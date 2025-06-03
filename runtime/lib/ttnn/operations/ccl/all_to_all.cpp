// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/all_to_all.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include <ttnn/distributed/api.hpp>

/*
TTNN does not yet expose an All-to-All collective as a first-class API,
and there is no point-to-point send/recv either.  To achieve the same
effect we fall back to the host:
1. Each device slices its local tensor shard.
2. The slices are copied to the host.
3. The host rearranges the slices so every device gets the piece it needs
   (a manual All-to-All shuffle).
4. The rearranged slices are copied back to the devices and concatenated.
*/
namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::AllToAllOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in());

  // ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();

  LOG_DEBUG("cos");
  auto tensor0 = ::ttnn::cos(inputTensor);
  LOG_DEBUG("get_device_tensors");
  auto deviceTensors = ::ttnn::distributed::get_device_tensors(tensor0);
  std::vector<std::shared_ptr<::ttnn::MeshDevice>> targetMeshes(
      deviceTensors.size());
  for (size_t idx = 0; idx < deviceTensors.size(); idx++) {
    targetMeshes[idx] =
        context.getUnitMeshDevice(deviceTensors.size() - idx - 1);
  }

  for (size_t idx = 0; idx < deviceTensors.size(); idx++) {
    LOG_DEBUG("TESTING IDX : ", idx);
    // auto submesh = meshDevice.create_submesh(::ttnn::MeshShape(1, 1),
    // ::ttnn::MeshCoordinate(0, 0));
    auto deviceTensor = deviceTensors[idx];
    LOG_DEBUG("from_device");
    auto hostTensor =
        ::ttnn::from_device(deviceTensor, true, ::ttnn::DefaultQueueId);
    LOG_DEBUG("to_device");
    auto movedTensor =
        ::ttnn::to_device(hostTensor, targetMeshes[idx].get(),
                          ::ttnn::DRAM_MEMORY_CONFIG, ::ttnn::DefaultQueueId);
  }
  LOG_DEBUG("End of Testing");

#if 0
  ::ttnn::MeshDevice *meshDevice = inputTensor.mesh_device();
  LOG_ASSERT(meshDevice != nullptr, "Tensor must belong to a mesh device");

  const int32_t splitDim = op->split_dim();
  const int32_t concatDim = op->concat_dim();
  const uint32_t splitCount = op->split_count();
  const uint32_t clusterAxis = op->cluster_axis();
  (void)concatDim;
  (void)clusterAxis;

  const auto meshShape = meshDevice->shape();
  const auto inputShape = inputTensor.logical_shape();
  LOG_ASSERT(inputShape[splitDim] % splitCount == 0,
             "Input dimension along splitDim must be divisible by splitCount");

  const uint32_t splitSize = inputShape[splitDim] / splitCount;
  std::vector<std::vector<::ttnn::Tensor>> gatheredTensorsSharded(
      splitCount, std::vector<::ttnn::Tensor>(meshDevice->num_devices()));
  auto steps = ::ttnn::SmallVector<int32_t>(inputShape.rank(), 1);
  for (size_t sliceIdx = 0; sliceIdx < splitCount; sliceIdx++) {
#if 0
    ::ttnn::SmallVector<int32_t> begins(inputShape.size(), 0);
    ::ttnn::SmallVector<int32_t> ends(inputShape.cbegin(), inputShape.cend());
    begins[splitDim] = sliceIdx * splitSize;
    ends[splitDim] = (sliceIdx + 1) * splitSize;
    ::ttnn::Tensor slice_device =
        ::ttnn::slice(::ttnn::DefaultQueueId, inputTensor, begins, ends, steps, ::ttnn::DRAM_MEMORY_CONFIG); // device
#else
  (void) splitSize;
  ::ttnn::Tensor slice_device = ::ttnn::cos(inputTensor);
  // ::ttnn::Tensor slice_device = inputTensor;
#endif
    std::vector<::ttnn::Tensor> deviceTensors =
        ::ttnn::distributed::get_device_tensors(slice_device);

    for (size_t idx = 0; idx < deviceTensors.size(); idx++) {
      LOG_DEBUG("TESTING IDX : ", idx);
      LOG_DEBUG("MeshDevice shape => (", meshDevice->shape()[0], ",", meshDevice->shape()[1], ")");

      LOG_DEBUG("Creating submesh");
      int32_t x = idx / 4;
      int32_t y = idx % 4;
      auto targetSubmesh = meshDevice->create_submesh(
          ::ttnn::MeshShape(1, 1), ::ttnn::MeshCoordinate(x, y));
      auto id = targetSubmesh->get_device(::ttnn::MeshCoordinate(0, 0))->id();
      LOG_DEBUG("target device ID :", id, " (", x, ",", y, ")");
      LOG_DEBUG("MeshDevice shape => (", meshDevice->shape()[0], ",", meshDevice->shape()[1], ")");
      LOG_DEBUG("targetSubmesh shape => (", targetSubmesh->shape()[0], ",", targetSubmesh->shape()[1], ")");

      LOG_DEBUG("Pulling tensor");
      ::ttnn::Tensor hostTensor = ::ttnn::from_device(deviceTensors[idx], true);
      LOG_DEBUG("Pulling tensor2");
      LOG_DEBUG("Pushing tensor");
      ::ttnn::Tensor deviceTensor =
          ::ttnn::to_device(hostTensor, targetSubmesh.get(), ::ttnn::DRAM_MEMORY_CONFIG);
      LOG_DEBUG("Done TESTING IDX : ", idx);
    }
    LOG_DEBUG("Done TESTING SLICE : ", sliceIdx);
  }

#if 0

  // 1. Slice the input tensor on-device and materialize per-device host copies.
  //    slicedTensorsMulti[sliceIdx][device_idx]
  std::vector<std::vector<::ttnn::Tensor>> slicedTensorsMulti(splitCount);

  for (uint32_t sliceIdx = 0; sliceIdx < splitCount; ++sliceIdx) {
    auto begins = ::ttnn::SmallVector<int32_t>(inputShape.rank(), 0);
    auto ends =
        ::ttnn::SmallVector<int32_t>(inputShape.cbegin(), inputShape.cend());
    auto steps = ::ttnn::SmallVector<int32_t>(inputShape.rank(), 1);

    begins[splitDim] = sliceIdx * splitSize;
    ends[splitDim] = (sliceIdx + 1) * splitSize;

    ::ttnn::Tensor slice_device =
        ::ttnn::slice(inputTensor, begins, ends, steps); // device
    slicedTensorsMulti[sliceIdx] = ::ttnn::distributed::get_device_tensors(
        ::ttnn::from_device(slice_device)); // host view
  }

  // 2. Transpose shards across devices inside each cluster.
  //    gatheredTensorsSharded[sliceIdx][device_idx]
  std::vector<std::vector<::ttnn::Tensor>> gatheredTensorsSharded(
      splitCount, std::vector<::ttnn::Tensor>(meshDevice->num_devices()));

  // (clusterId, DevId in the cluster) -> flat device index
  const auto clusterToDevIdx = [&](uint32_t clusterId,
                                   uint32_t innerDevId) -> uint32_t {
    uint32_t index;
    if (clusterAxis == 0) {
      index = innerDevId * meshShape[1] + clusterId;
    } else {
      index = clusterId * meshShape[1] + innerDevId;
    }
    return index;
  };

  for (uint32_t clusterId = 0; clusterId < meshShape[1 - clusterAxis];
       ++clusterId) {
    for (uint32_t srcRank = 0; srcRank < splitCount; ++srcRank) {
      const uint32_t srcDev = clusterToDevIdx(clusterId, srcRank);
      for (uint32_t sliceIdx = 0; sliceIdx < splitCount; ++sliceIdx) {
        const uint32_t dstDev = clusterToDevIdx(clusterId, sliceIdx);
        gatheredTensorsSharded[srcRank][dstDev] =
            slicedTensorsMulti[sliceIdx][srcDev];
      }
    }
  }

  // 3. Aggregate per-device shards back into device tensors and move them onto
  // the mesh.
  std::vector<::ttnn::Tensor> gatheredTensorsMulti(splitCount);
  for (uint32_t sliceIdx = 0; sliceIdx < splitCount; ++sliceIdx) {
    ::ttnn::Tensor shardedTensor = ::ttnn::distributed::aggregate_as_tensor(
        gatheredTensorsSharded[sliceIdx],
        inputTensor.distributed_tensor_config());

    gatheredTensorsMulti[sliceIdx] = ::ttnn::to_device(
        shardedTensor, meshDevice, inputTensor.memory_config());
  }

  // 4. Concatenate along the requested dimension and register output.
  ::ttnn::Tensor output = ::ttnn::concat(gatheredTensorsMulti, concatDim);
#endif
#endif
  LOG_DEBUG("End of AllToAll");
  ::ttnn::Tensor output = inputTensor;
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::ccl
