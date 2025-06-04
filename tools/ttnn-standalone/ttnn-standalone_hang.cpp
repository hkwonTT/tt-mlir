// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "hostdevcommon/common_values.hpp"

ttnn::Tensor testingHangFunction(ttnn::Tensor tensor, std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {
  ttnn::Tensor tensor0 = ttnn::cos(tensor);
  std::vector<ttnn::Tensor> deviceTensors = ttnn::distributed::get_device_tensors(tensor0);
  std::vector<ttnn::Tensor> reorgTensors(meshDevice->num_devices());
  for (size_t i = 0; i < deviceTensors.size(); i++)
  {
    auto hostTensor = ttnn::from_device(deviceTensors[i]);  // hangs here at second round
    auto targetDevice = meshDevice->create_submesh(ttnn::MeshShape(1, 1), ttnn::MeshCoordinate(0, deviceTensors.size() - i - 1));
    reorgTensors[targetDevice->build_id()] = ttnn::to_device(hostTensor, meshDevice.get(), std::nullopt);
  }
  ttnn::Tensor shardedTensor = ttnn::distributed::aggregate_as_tensor(reorgTensors, tensor.distributed_tensor_config());
  return shardedTensor;
}


ttnn::Tensor create_inputs_for_testing() {
  ttnn::Tensor v1 =
      ttnn::ones(ttnn::Shape({256, 256}), ttnn::DataType::FLOAT32,
                 ttnn::Layout::ROW_MAJOR, ::std::nullopt,
                 ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                                      ttnn::BufferType::SYSTEM_MEMORY});
  return v1;
}
std::shared_ptr<ttnn::distributed::MeshDevice> openMeshDevice() {
  return ttnn::distributed::open_mesh_device(
      ttnn::MeshShape(1, 2), DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE,
      1, tt::tt_metal::DispatchCoreConfig{tt::tt_metal::DispatchCoreType::ETH},
      std::nullopt, std::vector<int>{}, DEFAULT_WORKER_L1_SIZE);
}

ttnn::Tensor shardTensor(ttnn::Tensor inputTensor, std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {
  ttnn::distributed::Shard2dConfig shard2dConfig{std::nullopt, 1};
  ttnn::Tensor shardedInputHost = ttnn::distributed::distribute_tensor(
      inputTensor, *ttnn::distributed::shard_tensor_to_2d_mesh_mapper(
                       *meshDevice, meshDevice->shape(), shard2dConfig));

  ttnn::Tensor shardedInputLayout = ttnn::to_layout(
      shardedInputHost, ttnn::Layout::TILE, ::std::nullopt,
      ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                           ttnn::BufferType::SYSTEM_MEMORY},
      static_cast<ttnn::distributed::MeshDevice *>(nullptr));

  ttnn::Tensor shardedInput = ttnn::to_device(shardedInputLayout, meshDevice.get(),
      ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM});
  return shardedInput;
}

ttnn::Tensor unshardTensor(ttnn::Tensor shardedTensor, std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {
  ttnn::Tensor shardedHost = ttnn::from_device(shardedTensor);
  ttnn::Tensor shardedOutputLayout = ttnn::to_layout(
      shardedHost, ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                           ttnn::BufferType::SYSTEM_MEMORY},
      static_cast<ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::distributed::Concat2dConfig concat2dConfig{-1, 1};
  ttnn::Tensor outputTensor = ttnn::distributed::aggregate_tensor(
      shardedOutputLayout,
      *ttnn::distributed::concat_2d_mesh_to_tensor_composer(*meshDevice,
                                                              concat2dConfig));
  return outputTensor;
}
void testingHangCase(std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {
  ttnn::Tensor inputTensor = create_inputs_for_testing();
  ttnn::Tensor shardedInput = shardTensor(inputTensor, meshDevice);
  ttnn::Tensor shardedOutput = testingHangFunction(shardedInput, meshDevice);
  ttnn::Tensor outputTensor = unshardTensor(shardedOutput, meshDevice);
}

int32_t main() {
  std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice = openMeshDevice();
  testingHangCase(meshDevice);
  return 0;
}
