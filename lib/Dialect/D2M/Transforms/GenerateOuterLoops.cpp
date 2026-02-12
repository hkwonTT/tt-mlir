// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGENERATEOUTERLOOPS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MGenerateOuterLoopsRewriter : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  static std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
  buildLoopBounds(OpBuilder &builder, Location loc,
                  ArrayRef<int64_t> loopBounds) {
    Value zero = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                                   builder.getIndexAttr(0));
    Value one = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                                  builder.getIndexAttr(1));
    SmallVector<Value> lbs(loopBounds.size(), zero);
    SmallVector<Value> ubs(llvm::map_range(loopBounds, [&](int64_t dim) {
      return builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                               builder.getIndexAttr(dim));
    }));
    SmallVector<Value> step(loopBounds.size(), one);
    return std::make_tuple(lbs, ubs, step);
  }

  static scf::LoopNest buildLoopNest(PatternRewriter &rewriter, Location loc,
                                     ArrayRef<int64_t> loopBounds,
                                     Block *regionBlock, Block *loopedBlock) {
    auto [lbs, ubs, steps] = buildLoopBounds(rewriter, loc, loopBounds);

    return scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &bodyBuilder, Location loc, ValueRange iters) {
          rewriter.setInsertionPointToStart(bodyBuilder.getInsertionBlock());
          Block *innerLoopBlock = bodyBuilder.getInsertionBlock();
          rewriter.mergeBlocks(regionBlock, innerLoopBlock,
                               loopedBlock->getArguments());
        });
  }

  // If \p generic is inside a d2m.spatial, build phys_to_virt map from the
  // region's core range start so that physical core (sx,sy) maps to virtual
  // (0,0). Returns empty map when not in SpatialOp or start is (0,0). Uses
  // context's empty map to avoid uninitialized AffineMap.
  static AffineMap getPhysToVirtMapForSpatial(GenericOp generic) {
    MLIRContext *ctx = generic.getContext();
    auto emptyMap = AffineMap::get(ctx);

    auto *op = generic.getOperation();
    Operation *parent = op->getParentOp();
    d2m::SpatialOp spatialOp = nullptr;
    while (parent) {
      if (auto spatial = llvm::dyn_cast<d2m::SpatialOp>(parent)) {
        spatialOp = spatial;
        break;
      }
      parent = parent->getParentOp();
    }
    if (!spatialOp) {
      return emptyMap;
    }

    Region *genericRegion = op->getParentRegion();
    std::optional<unsigned> regionIndex;
    for (auto [idx, region] : llvm::enumerate(spatialOp.getRegions())) {
      if (genericRegion == &region) {
        regionIndex = idx;
        break;
      }
    }
    if (!regionIndex ||
        *regionIndex >= spatialOp.getGridRanges().getCoreRanges().size()) {
      return emptyMap;
    }

    ttcore::CoreRangeAttr coreRange =
        spatialOp.getGridRanges().getCoreRanges()[*regionIndex];
    int64_t startY = coreRange.getStartCoord().getY();
    int64_t startX = coreRange.getStartCoord().getX();
    if (startY == 0 && startX == 0) {
      return emptyMap;
    }

    // Map (d0, d1) -> (d0 - startY, d1 - startX): physical -> virtual.
    AffineExpr d0 = getAffineDimExpr(0, ctx);
    AffineExpr d1 = getAffineDimExpr(1, ctx);
    AffineExpr cY = getAffineConstantExpr(startY, ctx);
    AffineExpr cX = getAffineConstantExpr(startX, ctx);
    return AffineMap::get(2, 0, {d0 - cY, d1 - cX}, ctx);
  }

  static void replaceIndexOpUses(PatternRewriter &rewriter, Location loc,
                                 scf::LoopNest &loopNest, GenericOp generic) {
    // Get the output operand indexing map to determine which dimensions
    // participate in the grid
    unsigned outputOperandsIndex = generic.getOutputs().getBeginOperandIndex();
    AffineMap outputOperandIndexingMap =
        mlir::cast<AffineMapAttr>(
            generic.getIndexingMaps()[outputOperandsIndex])
            .getValue();

    // Get the grid mapping for use with CoreIndexOp. When this generic is
    // inside a d2m.spatial with a non-zero core range start, use a
    // physical-to-virtual map so core_index returns virtual indices.
    // Otherwise use the generic's grid mapping only when non-empty. Use a
    // flag to avoid touching an uninitialized AffineMap.
    AffineMap gridMapping = generic.getGrid().getMapping();
    AffineMap physToVirtMap = getPhysToVirtMapForSpatial(generic);
    bool usePhysToVirtMap = false;
    bool useGridMapping = false;
    if (!physToVirtMap.isEmpty()) {
      usePhysToVirtMap = true;
    } else if (!gridMapping.isEmpty()) {
      useGridMapping = true;
    }

    SmallVector<int64_t> blockFactors = generic.getBlockFactorsValue();

    // The number of grid dimensions (typically 2 for a 2D grid, but could be
    // more with virtualization). For spatial phys-to-virt map we have 2 dims.
    constexpr unsigned numPhysicalGridDims = 2;
    unsigned numGridDims = numPhysicalGridDims;
    if (usePhysToVirtMap) {
      numGridDims = physToVirtMap.getNumResults();
    } else if (useGridMapping) {
      numGridDims = gridMapping.getNumResults() - 1; // leading device result
    }

    // Create CoreIndexOp operations lazily - create them the first time we need
    // them, at the start of the outermost loop body, then reuse them.
    SmallVector<Value> virtualGridIndices;
    bool virtualGridIndicesCreated = false;

    // Handle BlockIndexOp: apply full grid/block calculation
    loopNest.loops.back().walk([&](BlockIndexOp index) {
      uint64_t dim = index.getDim();
      assert(dim < loopNest.loops.size());
      scf::ForOp loop = loopNest.loops[dim];
      Value iterIndex = loop.getInductionVar();

      // Set insertion point to before the BlockIndexOp so we can create
      // operations that will be used to replace it
      rewriter.setInsertionPoint(index);

      // Create CoreIndexOp operations lazily at the start of the outermost loop
      // body if we haven't created them yet. Use phys_to_virt map when present
      // (e.g. from d2m.spatial grid_ranges); otherwise use generic's grid
      // mapping. The lowering will apply the affine map.
      if (!virtualGridIndicesCreated && !loopNest.loops.empty()) {
        // Set insertion point to the start of the outermost loop body
        rewriter.setInsertionPointToStart(loopNest.loops.front().getBody());
        virtualGridIndices.resize(numGridDims);
        for (unsigned gridDim = 0; gridDim < numGridDims; gridDim++) {
          if (usePhysToVirtMap) {
            virtualGridIndices[gridDim] = rewriter.create<CoreIndexOp>(
                loc, static_cast<int64_t>(gridDim), physToVirtMap);
          } else if (useGridMapping) {
            virtualGridIndices[gridDim] = rewriter.create<CoreIndexOp>(
                loc, static_cast<int64_t>(gridDim), gridMapping);
          } else {
            virtualGridIndices[gridDim] = rewriter.create<CoreIndexOp>(
                loc, static_cast<int64_t>(gridDim));
          }
        }
        virtualGridIndicesCreated = true;
        // Reset insertion point back to before the BlockIndexOp
        rewriter.setInsertionPoint(index);
      }

      // Check if this iteration dimension maps to a grid dimension in the
      // output operand indexing map. The output operand indexing map maps
      // iteration dimensions to output dimensions. We need to check if the
      // expression corresponding to iteration dimension `dim` appears as one of
      // the first numGridDims results in the output map.
      //
      // Create the expression for iteration dimension `dim` (e.g., d0, d1, d2)
      AffineExpr dimExpr =
          getAffineDimExpr(dim, outputOperandIndexingMap.getContext());

      // Check if this expression appears in the output operand indexing map
      // results
      std::optional<unsigned> gridResult =
          outputOperandIndexingMap.getResultPosition(dimExpr);

      // If the result position exists and is less than numGridDims, then this
      // dimension participates in the grid
      if (gridResult.has_value() && gridResult.value() < numGridDims) {
        // This dimension participates in the grid. Compute:
        // gridIndex * blockFactor + iterIndex
        const unsigned gridDim = gridResult.value();
        assert(dim < blockFactors.size() && "Block factor index out of bounds");
        assert(gridDim < virtualGridIndices.size() &&
               "Grid dimension index out of bounds");

        Value gridIndex = virtualGridIndices[gridDim];
        Value blockFactor = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIndexType(),
            rewriter.getIndexAttr(blockFactors[dim]));

        Value gridScaled = rewriter.create<arith::MulIOp>(
            loc, rewriter.getIndexType(), gridIndex, blockFactor);
        Value combinedIndex = rewriter.create<arith::AddIOp>(
            loc, rewriter.getIndexType(), gridScaled, iterIndex);

        rewriter.replaceOp(index, combinedIndex);
      } else {
        // This dimension does not participate in the grid, just use the loop
        // induction variable
        rewriter.replaceOp(index, iterIndex);
      }
    });

    // Handle IterIndexOp: simple replacement with loop induction variable
    loopNest.loops.back().walk([&](IterIndexOp index) {
      uint64_t dim = index.getDim();
      assert(dim < loopNest.loops.size());
      scf::ForOp loop = loopNest.loops[dim];
      Value iterIndex = loop.getInductionVar();

      rewriter.setInsertionPoint(index);
      rewriter.replaceOp(index, iterIndex);
    });
  }

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const final {
    // Skip explicit datamovement form - users manage loops manually
    if (generic.isExplicitDatamovementForm()) {
      return failure();
    }

    // Only match GenericOp with a single region (single region/thread form
    // after DMA insertion)
    if (generic.getNumRegions() != 1) {
      return failure();
    }

    SmallVector<int64_t> loopBounds = generic.getLoopBounds();
    if (loopBounds.empty()) {
      // No loops to generate
      return failure();
    }

    // Check if loops are already generated (avoid infinite loop)
    // Look for the marker attribute on the outermost loop
    Region &checkRegion = generic.getRegion(0);
    if (!checkRegion.empty()) {
      Block &checkBlock = checkRegion.front();
      for (Operation &op : checkBlock.getOperations()) {
        if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
          if (forOp->hasAttr("d2m.outer_loop")) {
            return failure();
          }
        }
      }
    }

    // Create a new GenericOp with the same structure
    // After generating loops, preserve all attributes including block_factors
    // (needed by LowerLoadStoreOpsToDMA for stream index computation).
    auto loopedGeneric = rewriter.create<GenericOp>(
        generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
        generic.getOutputs(), generic.getGrid(),
        /* block_factors */ generic.getBlockFactors(),
        /* indexing_maps */ generic.getIndexingMaps(),
        /* iterator_types */ generic.getIteratorTypes(), generic.getThreads(),
        generic.getScratchInputsAttr(), generic.getNumRegions());

    // Process the single region
    Region &region = generic.getRegion(0);
    Block *regionBlock = &region.front();
    Block *loopedBlock = &loopedGeneric.getRegion(0).emplaceBlock();
    loopedBlock->addArguments(
        region.getArgumentTypes(),
        SmallVector<mlir::Location>(region.getArgumentTypes().size(),
                                    generic.getLoc()));
    rewriter.setInsertionPointToStart(loopedBlock);
    scf::LoopNest loopNest = buildLoopNest(
        rewriter, generic.getLoc(), loopBounds, regionBlock, loopedBlock);

    // Mark all loops in the nest with an attribute to prevent re-processing.
    // These are called "outer loops" because they wrap the generic operation,
    // iterating over its block factors. The generic operation's regions contain
    // the "inner" computation that executes within each loop iteration.
    for (scf::ForOp loop : loopNest.loops) {
      loop->setAttr("d2m.outer_loop", rewriter.getUnitAttr());
    }

    // Replace IterIndexOp uses. We need to do this after the loops are created
    // but before we replace the generic op, so the operations are created in
    // the right place. Set insertion point back to the start of loopedBlock
    // so CoreIndexOp operations are created in the right place.
    rewriter.setInsertionPointToStart(loopedBlock);
    replaceIndexOpUses(rewriter, generic.getLoc(), loopNest, loopedGeneric);

    rewriter.replaceOp(generic, loopedGeneric.getResults());

    return success();
  }
};
} // namespace

namespace {
class D2MGenerateOuterLoops
    : public impl::D2MGenerateOuterLoopsBase<D2MGenerateOuterLoops> {
public:
  using impl::D2MGenerateOuterLoopsBase<
      D2MGenerateOuterLoops>::D2MGenerateOuterLoopsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MGenerateOuterLoopsRewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
