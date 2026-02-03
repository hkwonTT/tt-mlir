// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/D2MToTTNN/D2MToTTNN.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt {

namespace {

// Resolve a GenericOp operand (stream_layout or cast) to the TTNN io value and
// the CB storage value. Used by both GenericOp and SpatialOp operand
// extraction.
static void resolveOperandToIoAndCb(Value operand, Value &io, Value &cb) {
  if (auto streamLayoutOp = mlir::dyn_cast_if_present<d2m::StreamLayoutOp>(
          operand.getDefiningOp());
      streamLayoutOp) {
    if (auto castOp = mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
            streamLayoutOp.getInput().getDefiningOp());
        castOp) {
      io = castOp.getOperand();
    } else {
      llvm_unreachable(
          "Expected TTNNMetalLayoutCastOp producing stream input.");
    }
    cb = streamLayoutOp.getStorage();
  } else if (auto castOp =
                 mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
                     operand.getDefiningOp());
             castOp) {
    io = castOp.getOperand();
    cb = operand;
  } else {
    llvm_unreachable("Expected stream_layout or cast op as operand.");
  }
}

static ttnn::ComputeKernelMathFidelity
convertMathFidelity(ttmetal::MathFidelity fidelity) {
  switch (fidelity) {
  case ttmetal::MathFidelity::LoFi:
    return ttnn::ComputeKernelMathFidelity::LoFi;
  case ttmetal::MathFidelity::HiFi2:
    return ttnn::ComputeKernelMathFidelity::HiFi2;
  case ttmetal::MathFidelity::HiFi3:
    return ttnn::ComputeKernelMathFidelity::HiFi3;
  case ttmetal::MathFidelity::HiFi4:
    return ttnn::ComputeKernelMathFidelity::HiFi4;
  }
  llvm_unreachable("Invalid MathFidelity");
}

class D2MGenericRewriter : public OpConversionPattern<d2m::GenericOp> {
public:
  D2MGenericRewriter(MLIRContext *context, ttmetal::MathFidelity mathFidelity)
      : OpConversionPattern<d2m::GenericOp>(context),
        mathFidelity(mathFidelity) {}

  static mlir::Attribute convertKernelArg(Builder &builder,
                                          const ttkernel::ArgAttr &arg) {
    switch (arg.getArgType()) {
    case ttkernel::ArgType::BufferAddress: {
      return builder.getAttr<ttnn::KernelArgAddressOfTensorAttr>(
          arg.getOperandIndex());
    }
    case ttkernel::ArgType::CBPort: {
      return builder.getAttr<ttnn::KernelArgCBBufferIndexAttr>(
          arg.getOperandIndex());
    }
    case ttkernel::ArgType::Semaphore: {
      return builder.getAttr<ttnn::KernelArgSemaphoreAtAttr>(
          arg.getOperandIndex());
    }
    }
  }

  static SmallVector<ttnn::KernelSemaphoreAttr>
  createSemaphoreDescriptors(Builder &builder, const ArrayAttr &threads,
                             const ttnn::CoreRangeSetAttr &coreRangeSet,
                             const SymbolTable &symbolTable) {
    llvm::DenseSet<size_t> seenSemaphoreIndices;

    for (Attribute threadAttr : threads) {
      auto thread = mlir::cast<d2m::ThreadAttr>(threadAttr);
      auto kernelFunc = symbolTable.lookup<func::FuncOp>(
          thread.getKernelSymbol().getRootReference());
      if (!kernelFunc) {
        continue;
      }

      auto kernelSpec = kernelFunc->getAttrOfType<ttkernel::ArgSpecAttr>(
          ttkernel::ArgSpecAttr::name);
      if (!kernelSpec) {
        continue;
      }

      for (auto ctArg : kernelSpec.getCtArgs()) {
        if (ctArg.getArgType() == ttkernel::ArgType::Semaphore) {
          seenSemaphoreIndices.insert(ctArg.getOperandIndex());
        }
      }
    }
    size_t numSemaphores = seenSemaphoreIndices.size();
    if (numSemaphores > 0) {
      // Semaphore indices are assigned sequentially in D2MToTTKernel, so they
      // should be dense.
      size_t minIndex = *llvm::min_element(seenSemaphoreIndices);
      size_t maxIndex = *llvm::max_element(seenSemaphoreIndices);
      TT_assertv((minIndex == 0u && maxIndex == numSemaphores - 1),
                 "Semaphore indices must be dense (0, 1, 2, ..., n-1)");
    }
    SmallVector<ttnn::KernelSemaphoreAttr> semaphoreDescriptors(numSemaphores);
    for (size_t i = 0; i < numSemaphores; ++i) {
      semaphoreDescriptors[i] = builder.getAttr<ttnn::KernelSemaphoreAttr>(
          /*id=*/i, ttnn::KernelCoreType::Worker, coreRangeSet,
          /*initial_value=*/0);
    }

    return semaphoreDescriptors;
  }

  static SmallVector<mlir::Attribute>
  createKernelDescriptors(Builder &builder, const ArrayAttr &threads,
                          const ttnn::CoreRangeSetAttr &coreRangeSet,
                          const SymbolTable &symbolTable,
                          ttmetal::MathFidelity mathFidelity) {
    SmallVector<mlir::Attribute> kernelConfigs(threads.size());
    int nocIndex = 0;
    for (const auto [i, thread] : llvm::enumerate(threads)) {
      const d2m::ThreadAttr threadAttr = mlir::cast<d2m::ThreadAttr>(thread);

      // Get kernel args.
      SymbolRefAttr kernelSymbol = threadAttr.getKernelSymbol();
      auto kernelFunc = symbolTable.lookup<mlir::func::FuncOp>(
          kernelSymbol.getRootReference());
      auto kernelSpec = kernelFunc->getAttrOfType<ttkernel::ArgSpecAttr>(
          ttkernel::ArgSpecAttr::name);

      // Note: D2MToTTKernel will only populate kernelSpec with rtargs in the
      // ttnn-mode, however despite the name, they are actually common runtime
      // args. TTKernel ArgSpec does not have crt field, and the normal tt-metal
      // path doesn't use rt args at all.
      auto crtArgs = kernelSpec.getRtArgs();
      auto ctArgs = kernelSpec.getCtArgs();
      llvm::SmallVector<mlir::Attribute> kernelCTArgs(ctArgs.size());
      llvm::SmallVector<mlir::Attribute> kernelCRTArgs(crtArgs.size());
      for (const auto [i, arg] : llvm::enumerate(crtArgs)) {
        kernelCRTArgs[i] = convertKernelArg(builder, arg);
      }
      for (const auto [i, arg] : llvm::enumerate(ctArgs)) {
        kernelCTArgs[i] = convertKernelArg(builder, arg);
      }

      // Create KernelDescriptor.
      switch (threadAttr.getThreadType()) {
      case d2m::ThreadType::Compute: {
        // TODO (vtangTT) #5032: support lowering to different compute configs.
        kernelConfigs[i] = builder.getAttr<ttnn::ComputeKernelAttr>(
            kernelSymbol, coreRangeSet,
            /*math_fidelity*/ convertMathFidelity(mathFidelity),
            /*fp32DestAccum*/ false,
            /*dst_full_sync_en*/ false,
            /*unpack_to_dest_mode*/
            ArrayRef<ttnn::ComputeKernelUnpackToDestMode>{
                ttnn::ComputeKernelUnpackToDestMode::Default},
            /*bfp8_pack_precise*/ false,
            /*math_approx_mode*/ false, kernelCRTArgs, kernelCTArgs);
        break;
      }
      // TODO (vtangTT) #5033: fix this assumption that order is
      // read->write->compute; nocIndex == 0 for read, nocIndex == 1 for write.
      case d2m::ThreadType::Datamovement: {
        TT_assert(nocIndex < 2);
        if (nocIndex == 0) {
          kernelConfigs[i] = builder.getAttr<ttnn::ReadKernelAttr>(
              kernelSymbol, coreRangeSet, kernelCRTArgs, kernelCTArgs);
        } else {
          kernelConfigs[i] = builder.getAttr<ttnn::WriteKernelAttr>(
              kernelSymbol, coreRangeSet, kernelCRTArgs, kernelCTArgs);
        }
        nocIndex++;
        break;
      }
      case d2m::ThreadType::Unified: {
        // Unified threads should have been split by SplitUnifiedThread before
        // reaching this pass.
        llvm_unreachable("Unexpected thread type in backend conversion");
      }
      }
    }
    return kernelConfigs;
  }

  static SmallVector<ttnn::KernelCBAttr>
  createCBDescriptors(Builder &builder, const llvm::SmallVector<Value> &cbs,
                      const ttcore::DeviceAttr &device,
                      const ttnn::CoreRangeSetAttr &coreRangeSet) {
    if (cbs.empty()) {
      llvm_unreachable("Expected circular buffers.");
    }

    MLIRContext *ctx = builder.getContext();
    llvm::SmallVector<ttnn::KernelCBAttr> cbDescriptors(cbs.size());

    for (auto [i, cb] : llvm::enumerate(cbs)) {
      auto cb_memref = dyn_cast<MemRefType>(cb.getType());
      TT_assertv(mlir::isa<ttcore::TileType>(cb_memref.getElementType()),
                 "Only TileType supported.");
      ttcore::DataType dtype =
          ttcore::elementTypeToDataType(cb_memref.getElementType());
      size_t pageSize = device.getMemrefCBPageSizeBytes(cb_memref);
      size_t numPages = device.getMemrefCBNumPages(cb_memref);

      ttnn::KernelCBFormatAttr cbFormat =
          ttnn::KernelCBFormatAttr::get(ctx, i, dtype, pageSize);

      ttnn::KernelCBGlobalBufferAddressOfTensorAttr globalCBIndexOfTensor;
      if (auto castOp = mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
              cb.getDefiningOp())) {
        // Input is not streamed, thus buffer must be aliased.
        TT_assertv(ttcore::getMemorySpace(cb_memref) ==
                       ttcore::MemorySpace::DeviceL1,
                   "Can only alias L1 buffers.");
        globalCBIndexOfTensor =
            ttnn::KernelCBGlobalBufferAddressOfTensorAttr::get(ctx, i);
      }
      cbDescriptors[i] =
          ttnn::KernelCBAttr::get(ctx, numPages * pageSize, coreRangeSet,
                                  {cbFormat}, globalCBIndexOfTensor);
    }

    return cbDescriptors;
  }

  static ttnn::CoreRangeSetAttr createCoreRangeSet(
      mlir::Builder &builder, llvm::ArrayRef<int64_t> gridSize,
      std::optional<llvm::ArrayRef<int64_t>> startCoord = std::nullopt) {

    llvm::SmallVector<int64_t, 4> defaultStart;
    llvm::ArrayRef<int64_t> startCoordRef;

    if (startCoord.has_value()) {
      startCoordRef = *startCoord;
    } else {
      defaultStart.assign(gridSize.size(), 0);
      startCoordRef = defaultStart;
    }

    return ttnn::CoreRangeSetAttr::get(
        builder.getContext(),
        ttnn::CoreRangeAttr::get(
            builder.getContext(),
            ttnn::CoreCoordAttr::get(builder.getContext(), startCoordRef[0],
                                     startCoordRef[1]),
            ttnn::CoreCoordAttr::get(builder.getContext(),
                                     startCoordRef[0] + gridSize[0] - 1,
                                     startCoordRef[1] + gridSize[1] - 1)));
  }

  // Structure to hold all descriptors and I/O values for a GenericOp.
  struct GenericOpDescriptors {
    SmallVector<mlir::Attribute> kernelDescriptors;
    SmallVector<ttnn::KernelCBAttr> cbDescriptors;
    SmallVector<ttnn::KernelSemaphoreAttr> semaphoreDescriptors;
    SmallVector<Value> ios;
  };

  // Extract I/O operands and circular buffers from a GenericOp.
  static void extractOperandsFromGenericOp(d2m::GenericOp op,
                                           llvm::SmallVector<Value> &ios,
                                           llvm::SmallVector<Value> &cbs) {
    const size_t size = op.getOperands().size();
    ios.resize(size);
    cbs.resize(size);
    for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
      resolveOperandToIoAndCb(operand, ios[i], cbs[i]);
    }
  }

  // Extract inputs and outputs separately from a GenericOp.
  static void
  extractInputsAndOutputsFromGenericOp(d2m::GenericOp op,
                                       llvm::SmallVector<Value> &inputs,
                                       llvm::SmallVector<Value> &outputs) {
    for (Value input : op.getInputs()) {
      Value ioValue;
      Value cbValue;
      resolveOperandToIoAndCb(input, ioValue, cbValue);
      inputs.push_back(ioValue);
    }
    for (Value output : op.getOutputs()) {
      Value ioValue;
      Value cbValue;
      resolveOperandToIoAndCb(output, ioValue, cbValue);
      outputs.push_back(ioValue);
    }
  }

  // Compute grid size from a GenericOp.
  static llvm::SmallVector<int64_t>
  computeGridSizeFromGenericOp(d2m::GenericOp op) {
    ttcore::GridAttr opGrid = op.getGrid();
    llvm::SmallVector<int64_t> gridSize;

    if (!opGrid.getMapping().isEmpty()) {
      // The genericOp has a virtual grid. We need to recover the original
      // physical grid.
      auto output = op.getOutputs()[0];
      mlir::ShapedType outputType =
          mlir::cast<mlir::ShapedType>(output.getType());
      auto shardLayout = mlir::dyn_cast<ttcore::ShardLayoutAttr>(
          ttcore::getDeviceLayout(outputType));
      TT_assertv(shardLayout, "Expected shardLayoutAttr for the output of a "
                              "generic op with a virtual grid.");

      auto physicalGridShape = d2m::utils::getPhysicalGridShape(output);
      // TTNN grids are (Width, Height), while D2M grids are (Height, Width).
      gridSize = {physicalGridShape[1], physicalGridShape[0]};
    } else {
      // TTNN grids are (Width, Height), while D2M grids are (Height, Width).
      gridSize = {opGrid.getShape()[1], opGrid.getShape()[0]};
    }

    return gridSize;
  }

  // Create all descriptors from a GenericOp. Grid size is computed from the
  // op; startCoord is only used for SpatialOp (region placement). GenericOp
  // uses default (0,0) when startCoord is std::nullopt.
  static GenericOpDescriptors createDescriptorsFromGenericOp(
      Builder &builder, d2m::GenericOp op, const ttcore::DeviceAttr &device,
      std::optional<llvm::ArrayRef<int64_t>> startCoord,
      const SymbolTable &symbolTable, ttmetal::MathFidelity mathFidelity) {
    llvm::SmallVector<int64_t> gridSize = computeGridSizeFromGenericOp(op);
    ttnn::CoreRangeSetAttr coreRangeSet =
        createCoreRangeSet(builder, gridSize, startCoord);

    GenericOpDescriptors descriptors;

    // Extract operands (ios and cbs).
    llvm::SmallVector<Value> cbs;
    extractOperandsFromGenericOp(op, descriptors.ios, cbs);

    // Create CB descriptors.
    descriptors.cbDescriptors =
        createCBDescriptors(builder, cbs, device, coreRangeSet);

    // Create KernelDescriptors.
    descriptors.kernelDescriptors = createKernelDescriptors(
        builder, op.getThreads(), coreRangeSet, symbolTable, mathFidelity);

    // Extract semaphore descriptors from kernel functions.
    descriptors.semaphoreDescriptors = createSemaphoreDescriptors(
        builder, op.getThreads(), coreRangeSet, symbolTable);

    return descriptors;
  }

  // Create a ttnn::GenericOp from descriptors.
  static ttnn::GenericOp
  createTTNNGenericOpFromDescriptors(ConversionPatternRewriter &rewriter,
                                     Operation *op,
                                     const GenericOpDescriptors &descriptors) {
    MLIRContext *ctx = rewriter.getContext();

    ttnn::ProgramAttr program = ttnn::ProgramAttr::get(
        ctx, descriptors.kernelDescriptors, descriptors.cbDescriptors,
        descriptors.semaphoreDescriptors);

    return rewriter.create<ttnn::GenericOp>(op->getLoc(), descriptors.ios,
                                            program, ttnn::MemoryConfigAttr());
  }

  LogicalResult
  matchAndRewrite(d2m::GenericOp op, d2m::GenericOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto device = ttcore::lookupDevice(op->getParentOp());
    TT_assert(device);

    SymbolTable opSymTable(op->getParentOfType<ModuleOp>());
    GenericOpDescriptors descriptors = createDescriptorsFromGenericOp(
        rewriter, op, device, std::nullopt, opSymTable, this->mathFidelity);

    // Create ttnn::GenericOp and replace.
    auto ttnnGenericOp = createTTNNGenericOpFromDescriptors(
        rewriter, op.getOperation(), descriptors);
    rewriter.replaceOp(op, ttnnGenericOp->getResults());

    return success();
  };

private:
  ttmetal::MathFidelity mathFidelity;
};
} // namespace

namespace {
class D2MSpatialRewriter : public OpConversionPattern<d2m::SpatialOp> {
public:
  D2MSpatialRewriter(MLIRContext *context, ttmetal::MathFidelity mathFidelity)
      : OpConversionPattern<d2m::SpatialOp>(context),
        mathFidelity(mathFidelity) {}

  // Merge all regions of a SpatialOp into one GenericOpDescriptors (unique
  // I/O, concatenated descriptors with remapped CB/tensor indices).
  static LogicalResult mergeSpatialOpToDescriptors(
      ConversionPatternRewriter &rewriter, d2m::SpatialOp op,
      const ttcore::DeviceAttr &device, const SymbolTable &symbolTable,
      ttmetal::MathFidelity mathFidelity,
      D2MGenericRewriter::GenericOpDescriptors &mergedDescriptors) {
    ttcore::CoreRangeSetAttr gridRanges = op.getGridRanges();
    auto coreRanges = gridRanges.getCoreRanges();
    TT_assert(!coreRanges.empty());
    TT_assert((op->getRegions().size() == coreRanges.size() &&
               "SpatialOp region count must match grid_ranges size"));

    llvm::SetVector<Value> allInputs;
    llvm::SetVector<Value> allOutputs;

    for (Region &region : op->getRegions()) {
      auto genericOpIt = region.front().getOps<d2m::GenericOp>().begin();
      if (genericOpIt == region.front().getOps<d2m::GenericOp>().end()) {
        return rewriter.notifyMatchFailure(
            op, "Expected a GenericOp in each region of SpatialOp");
      }
      d2m::GenericOp genericOp = *genericOpIt;
      llvm::SmallVector<Value> regionInputs;
      llvm::SmallVector<Value> regionOutputs;
      D2MGenericRewriter::extractInputsAndOutputsFromGenericOp(
          genericOp, regionInputs, regionOutputs);

      allInputs.insert(regionInputs.begin(), regionInputs.end());
      allOutputs.insert(regionOutputs.begin(), regionOutputs.end());
    }

    llvm::DenseMap<Value, size_t> valueToIosIndex;
    for (auto [i, v] : llvm::enumerate(allInputs)) {
      valueToIosIndex[v] = i;
    }
    for (auto [i, v] : llvm::enumerate(allOutputs)) {
      valueToIosIndex[v] = allInputs.size() + i;
    }

    SmallVector<mlir::Attribute> allKernelDescriptors;
    SmallVector<ttnn::KernelCBAttr> allCBDescriptors;
    SmallVector<ttnn::KernelSemaphoreAttr> allSemaphoreDescriptors;

    size_t regionIndex = 0;
    for (Region &region : op->getRegions()) {
      d2m::GenericOp genericOp =
          *region.front().getOps<d2m::GenericOp>().begin();

      llvm::SmallVector<size_t> localToGlobalTensorIndex =
          computeLocalToGlobalTensorIndex(genericOp, valueToIosIndex);

      TT_assert(regionIndex < coreRanges.size());
      auto coreRange = coreRanges[regionIndex];
      auto startCoord = coreRange.getStartCoord();
      llvm::SmallVector<int64_t> startCoordVec = {startCoord.getX(),
                                                  startCoord.getY()};

      D2MGenericRewriter::GenericOpDescriptors regionDescriptors =
          D2MGenericRewriter::createDescriptorsFromGenericOp(
              rewriter, genericOp, device,
              std::optional<llvm::ArrayRef<int64_t>>(startCoordVec),
              symbolTable, mathFidelity);

      appendRegionDescriptors(rewriter, regionDescriptors,
                              allCBDescriptors.size(), localToGlobalTensorIndex,
                              allKernelDescriptors, allCBDescriptors,
                              allSemaphoreDescriptors);

      regionIndex++;
    }

    mergedDescriptors.kernelDescriptors = std::move(allKernelDescriptors);
    mergedDescriptors.cbDescriptors = std::move(allCBDescriptors);
    mergedDescriptors.semaphoreDescriptors = std::move(allSemaphoreDescriptors);
    mergedDescriptors.ios.assign(allInputs.begin(), allInputs.end());
    mergedDescriptors.ios.append(allOutputs.begin(), allOutputs.end());
    return success();
  }

  LogicalResult
  matchAndRewrite(d2m::SpatialOp op, d2m::SpatialOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto device = ttcore::lookupDevice(op->getParentOp());
    TT_assert(device);

    SymbolTable opSymTable(op->getParentOfType<ModuleOp>());
    D2MGenericRewriter::GenericOpDescriptors mergedDescriptors;
    if (failed(mergeSpatialOpToDescriptors(rewriter, op, device, opSymTable,
                                           this->mathFidelity,
                                           mergedDescriptors))) {
      return failure();
    }

    auto ttnnGenericOp = D2MGenericRewriter::createTTNNGenericOpFromDescriptors(
        rewriter, op.getOperation(), mergedDescriptors);
    rewriter.replaceOp(op, ttnnGenericOp->getResults());
    return success();
  };

private:
  // Map a GenericOp's operands (inputs then outputs) to global I/O indices
  // for the merged SpatialOp I/O list.
  static llvm::SmallVector<size_t> computeLocalToGlobalTensorIndex(
      d2m::GenericOp op, const llvm::DenseMap<Value, size_t> &valueToIosIndex) {
    llvm::SmallVector<size_t> localToGlobal;
    for (Value input : op.getInputs()) {
      Value ioValue;
      Value cbValue;
      resolveOperandToIoAndCb(input, ioValue, cbValue);
      auto it = valueToIosIndex.find(ioValue);
      TT_assert(it != valueToIosIndex.end());
      localToGlobal.push_back(it->second);
    }
    for (Value output : op.getOutputs()) {
      Value ioValue;
      Value cbValue;
      resolveOperandToIoAndCb(output, ioValue, cbValue);
      auto it = valueToIosIndex.find(ioValue);
      TT_assert(it != valueToIosIndex.end());
      localToGlobal.push_back(it->second);
    }
    return localToGlobal;
  }

  // Append one region's descriptors with remapping (CB index offset and
  // local->global tensor indices). Used when merging SpatialOp regions.
  static void appendRegionDescriptors(
      ConversionPatternRewriter &rewriter,
      const D2MGenericRewriter::GenericOpDescriptors &regionDescriptors,
      size_t cbIndexOffset, llvm::ArrayRef<size_t> localToGlobalTensorIndex,
      SmallVector<mlir::Attribute> &allKernelDescriptors,
      SmallVector<ttnn::KernelCBAttr> &allCBDescriptors,
      SmallVector<ttnn::KernelSemaphoreAttr> &allSemaphoreDescriptors) {
    for (mlir::Attribute kernelAttr : regionDescriptors.kernelDescriptors) {
      allKernelDescriptors.push_back(remapKernelDescriptorForSpatial(
          kernelAttr, cbIndexOffset, localToGlobalTensorIndex));
    }
    for (auto cbDesc : regionDescriptors.cbDescriptors) {
      SmallVector<ttnn::KernelCBFormatAttr> remappedFormats;
      for (auto format : cbDesc.getFormats()) {
        uint32_t newBufferIndex = format.getBufferIndex() + cbIndexOffset;
        remappedFormats.push_back(ttnn::KernelCBFormatAttr::get(
            rewriter.getContext(), newBufferIndex, format.getDtype(),
            format.getPageSize()));
      }
      ttnn::KernelCBGlobalBufferAddressOfTensorAttr remappedGlobalBuffer;
      if (auto globalBuffer = cbDesc.getBuffer()) {
        remappedGlobalBuffer =
            ttnn::KernelCBGlobalBufferAddressOfTensorAttr::get(
                rewriter.getContext(), globalBuffer.getTensorOperandIndex());
      }
      allCBDescriptors.push_back(ttnn::KernelCBAttr::get(
          rewriter.getContext(), cbDesc.getTotalSize(), cbDesc.getCoreRanges(),
          remappedFormats, remappedGlobalBuffer));
    }
    allSemaphoreDescriptors.append(
        regionDescriptors.semaphoreDescriptors.begin(),
        regionDescriptors.semaphoreDescriptors.end());
  }

  // Remap kernel args when merging regions: CB buffer index (add offset) and
  // address-of-tensor (local operand index -> global allIos).
  static SmallVector<mlir::Attribute>
  remapKernelArgsForSpatial(MLIRContext *ctx, ArrayRef<mlir::Attribute> args,
                            size_t cbIndexOffset,
                            ArrayRef<size_t> localToGlobalTensorIndex) {
    SmallVector<mlir::Attribute> remapped;
    for (mlir::Attribute arg : args) {
      if (auto cbArg = mlir::dyn_cast<ttnn::KernelArgCBBufferIndexAttr>(arg)) {
        size_t newIndex = cbArg.getBufferIndex() + cbIndexOffset;
        remapped.push_back(
            ttnn::KernelArgCBBufferIndexAttr::get(ctx, newIndex));
      } else if (auto tensorArg =
                     mlir::dyn_cast<ttnn::KernelArgAddressOfTensorAttr>(arg)) {
        size_t localIndex = tensorArg.getTensorIndex();
        TT_assert(localIndex < localToGlobalTensorIndex.size());
        size_t globalIndex = localToGlobalTensorIndex[localIndex];
        remapped.push_back(
            ttnn::KernelArgAddressOfTensorAttr::get(ctx, globalIndex));
      } else {
        remapped.push_back(arg);
      }
    }
    return remapped;
  }

  // Remap a kernel descriptor for SpatialOp: CB indices and
  // #ttnn.kernel_arg_address_of_tensor (local -> global allIos index).
  static mlir::Attribute
  remapKernelDescriptorForSpatial(mlir::Attribute kernelAttr,
                                  size_t cbIndexOffset,
                                  ArrayRef<size_t> localToGlobalTensorIndex) {
    MLIRContext *ctx = kernelAttr.getContext();

    if (auto computeKernel =
            mlir::dyn_cast<ttnn::ComputeKernelAttr>(kernelAttr)) {
      auto ctArgs =
          remapKernelArgsForSpatial(ctx, computeKernel.getCtArgs(),
                                    cbIndexOffset, localToGlobalTensorIndex);
      auto commonRtArgs =
          remapKernelArgsForSpatial(ctx, computeKernel.getCommonRtArgs(),
                                    cbIndexOffset, localToGlobalTensorIndex);
      return ttnn::ComputeKernelAttr::get(
          ctx, computeKernel.getSymbolRef(), computeKernel.getCoreRanges(),
          computeKernel.getMathFidelity(), computeKernel.getFp32DestAccEn(),
          computeKernel.getDstFullSyncEn(),
          computeKernel.getUnpackToDestModes(),
          computeKernel.getBfp8PackPrecise(), computeKernel.getMathApproxMode(),
          commonRtArgs, computeKernel.getRtArgs(), ctArgs);
    }

    if (auto readKernel = mlir::dyn_cast<ttnn::ReadKernelAttr>(kernelAttr)) {
      auto ctArgs = remapKernelArgsForSpatial(
          ctx, readKernel.getCtArgs(), cbIndexOffset, localToGlobalTensorIndex);
      auto commonRtArgs =
          remapKernelArgsForSpatial(ctx, readKernel.getCommonRtArgs(),
                                    cbIndexOffset, localToGlobalTensorIndex);
      return ttnn::ReadKernelAttr::get(ctx, readKernel.getSymbolRef(),
                                       readKernel.getCoreRanges(), commonRtArgs,
                                       readKernel.getRtArgs(), ctArgs);
    }

    if (auto writeKernel = mlir::dyn_cast<ttnn::WriteKernelAttr>(kernelAttr)) {
      auto ctArgs =
          remapKernelArgsForSpatial(ctx, writeKernel.getCtArgs(), cbIndexOffset,
                                    localToGlobalTensorIndex);
      auto commonRtArgs =
          remapKernelArgsForSpatial(ctx, writeKernel.getCommonRtArgs(),
                                    cbIndexOffset, localToGlobalTensorIndex);
      return ttnn::WriteKernelAttr::get(
          ctx, writeKernel.getSymbolRef(), writeKernel.getCoreRanges(),
          commonRtArgs, writeKernel.getRtArgs(), ctArgs);
    }

    return kernelAttr;
  }

  ttmetal::MathFidelity mathFidelity;
};
} // namespace

namespace {
class TTNNMetalLayoutCastRewriter
    : public OpConversionPattern<ttir::TTNNMetalLayoutCastOp> {
public:
  using OpConversionPattern<ttir::TTNNMetalLayoutCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::TTNNMetalLayoutCastOp op,
                  ttir::TTNNMetalLayoutCastOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (auto inner =
            op.getOperand().getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
      rewriter.replaceOp(op, inner.getOperand());
    } else if (auto inner =
                   op.getOperand().getDefiningOp<d2m::StreamLayoutOp>()) {
      // Match the pattern cast(stream(cast(output_tensor))) and rewrite as just
      // output_tensor.
      if (auto inner2 =
              inner.getInput().getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
        rewriter.replaceOp(op, inner2.getOperand());
      }
    }
    return success();
  };
};
} // namespace

namespace {
class StreamLayoutRewriter : public OpConversionPattern<d2m::StreamLayoutOp> {
public:
  using OpConversionPattern<d2m::StreamLayoutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::StreamLayoutOp op, d2m::StreamLayoutOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  };
};
} // namespace

namespace {
class D2MEmptyRewriter : public OpConversionPattern<d2m::EmptyOp> {
public:
  using OpConversionPattern<d2m::EmptyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::EmptyOp op, d2m::EmptyOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *ctx = rewriter.getContext();
    auto tensorType = cast<RankedTensorType>(op.getResult().getType());
    auto encoding = tensorType.getEncoding();
    auto shape = ttnn::ShapeAttr::get(ctx, tensorType.getShape());

    ttcore::DataTypeAttr dtype;
    ttnn::LayoutAttr layout;
    ttnn::MemoryConfigAttr memcfg;

    // Reuses the existing ttnn.get_device op if present, else create one.
    auto device = ttnn::utils::getOrInsertDevice(rewriter, op);
    auto deviceAttr = ttcore::lookupDevice(op);

    // Handle both TTNNLayoutAttr and TTNNNDLayoutAttr
    if (auto layoutAttr = mlir::dyn_cast<ttnn::TTNNLayoutAttr>(encoding)) {
      dtype = ttcore::DataTypeAttr::get(ctx, layoutAttr.getDataType());
      layout = ttnn::LayoutAttr::get(ctx, layoutAttr.getLayout());
      memcfg =
          ttnn::MemoryConfigAttr::get(layoutAttr, deviceAttr.getWorkerGrid());
    } else if (auto ndLayoutAttr =
                   mlir::dyn_cast<ttnn::TTNNNDLayoutAttr>(encoding)) {
      dtype = ttcore::DataTypeAttr::get(ctx, ndLayoutAttr.getDataType());
      layout = ttnn::LayoutAttr::get(ctx, ndLayoutAttr.getLayout());
      auto bufferType =
          ttnn::BufferTypeAttr::get(ctx, ndLayoutAttr.getBufferType());
      auto ndShardSpec = ttnn::NDShardSpecAttr::get(ndLayoutAttr);
      memcfg = ttnn::MemoryConfigAttr::get(
          ctx, ndLayoutAttr.getMemLayout(), bufferType,
          /*shardSpec=*/std::nullopt, ndShardSpec);
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported encoding type");
    }

    rewriter.replaceOpWithNewOp<ttnn::EmptyOp>(op, tensorType, device, shape,
                                               dtype, layout, memcfg);
    return success();
  };
};
} // namespace

namespace {
class D2MFullRewriter : public OpConversionPattern<d2m::FullOp> {
public:
  using OpConversionPattern<d2m::FullOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::FullOp op, d2m::FullOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *ctx = rewriter.getContext();
    auto tensorType = cast<RankedTensorType>(op.getResult().getType());
    auto encoding = tensorType.getEncoding();

    // Convert DenseI32ArrayAttr shape to ttnn::ShapeAttr
    auto shapeI32 = adaptor.getShape();
    SmallVector<int64_t> shapeI64(shapeI32.begin(), shapeI32.end());
    auto shape = ttnn::ShapeAttr::get(ctx, shapeI64);

    ttcore::DataTypeAttr dtype;
    ttnn::LayoutAttr layout;
    ttnn::MemoryConfigAttr memcfg;

    // Reuses the existing ttnn.get_device op if present, else create one.
    auto device = ttnn::utils::getOrInsertDevice(rewriter, op);
    auto deviceAttr = ttcore::lookupDevice(op);

    // Handle both TTNNLayoutAttr and TTNNNDLayoutAttr
    if (auto layoutAttr = mlir::dyn_cast<ttnn::TTNNLayoutAttr>(encoding)) {
      dtype = ttcore::DataTypeAttr::get(ctx, layoutAttr.getDataType());
      layout = ttnn::LayoutAttr::get(ctx, layoutAttr.getLayout());
      memcfg =
          ttnn::MemoryConfigAttr::get(layoutAttr, deviceAttr.getWorkerGrid());
    } else if (auto ndLayoutAttr =
                   mlir::dyn_cast<ttnn::TTNNNDLayoutAttr>(encoding)) {
      dtype = ttcore::DataTypeAttr::get(ctx, ndLayoutAttr.getDataType());
      layout = ttnn::LayoutAttr::get(ctx, ndLayoutAttr.getLayout());
      auto bufferType =
          ttnn::BufferTypeAttr::get(ctx, ndLayoutAttr.getBufferType());
      auto ndShardSpec = ttnn::NDShardSpecAttr::get(ndLayoutAttr);
      memcfg = ttnn::MemoryConfigAttr::get(
          ctx, ndLayoutAttr.getMemLayout(), bufferType,
          /*shardSpec=*/std::nullopt, ndShardSpec);
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported encoding type");
    }

    rewriter.replaceOpWithNewOp<ttnn::FullOp>(op, tensorType, device, shape,
                                              adaptor.getFillValue(), dtype,
                                              layout, memcfg);
    return success();
  };
};
} // namespace

void populateD2MToTTNNPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                               TypeConverter &typeConverter,
                               ttmetal::MathFidelity mathFidelity) {
  patterns.add<D2MGenericRewriter, D2MSpatialRewriter>(ctx, mathFidelity);
  patterns.add<TTNNMetalLayoutCastRewriter, D2MEmptyRewriter, D2MFullRewriter,
               StreamLayoutRewriter>(ctx);
}
} // namespace mlir::tt
