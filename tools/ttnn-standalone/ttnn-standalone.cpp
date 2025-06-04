// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "hostdevcommon/common_values.hpp"
#include <iostream>
#include <stdexcept>


ttnn::Tensor testingFunction1(ttnn::Tensor shardedInput, std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {

  ttnn::Tensor tensor0 = ttnn::cos(shardedInput);

  std::cout << "Retrieving single device tensors from sharded multi device tensor" << std::endl;
  std::vector<ttnn::Tensor> deviceTensors = ttnn::distributed::get_device_tensors(tensor0);
  std::vector<ttnn::Tensor> reorgTensors(deviceTensors.size());
  for (size_t idx = 0; idx < deviceTensors.size(); idx++) {
    std::cout << "Reorganizing tensor from Device #" << idx << std::endl;
    auto &deviceTensor = deviceTensors[idx];
    std::cout << " - Pulling tensor from Device #" << idx << std::endl;
    auto hostTensor = ttnn::from_device(deviceTensor, true, ttnn::DefaultQueueId);
    std::cout << " - Creating unit submesh of Device #" << (1 - idx) << std::endl;
    auto submesh = meshDevice->create_submesh(ttnn::MeshShape(1, 1), ttnn::MeshCoordinate(0, 1 - idx));
    std::cout << " - Pushing tensor to Device #" << (1 - idx) << std::endl;
    reorgTensors[1 - idx] = ttnn::to_device(hostTensor, submesh.get(), ttnn::DRAM_MEMORY_CONFIG, ttnn::DefaultQueueId);
  }
  std::cout << "Aggregating single device tensors as a sharded multi device Tensor" << std::endl;
  ttnn::Tensor shardedTensor = ttnn::distributed::aggregate_as_tensor(reorgTensors, shardedInput.distributed_tensor_config());

  return shardedTensor;
}


ttnn::Tensor testingFunction2(ttnn::Tensor shardedInput, std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {

  ttnn::Tensor tensor0 = ttnn::cos(shardedInput);

  std::cout << "Retrieving single device tensors from sharded multi device tensor" << std::endl;
  std::vector<ttnn::Tensor> deviceTensors = ttnn::distributed::get_device_tensors(tensor0);

  std::vector<ttnn::Tensor> reorgTensors(deviceTensors.size());
  {
    std::vector<std::shared_ptr<ttnn::distributed::MeshDevice>> targetSubmeshes;
    for (size_t idx = 0; idx < deviceTensors.size(); idx++) {
      std::cout << " - Creating unit submesh of Device #" << (1 - idx) << std::endl;
      targetSubmeshes.push_back(meshDevice->create_submesh(ttnn::MeshShape(1, 1), ttnn::MeshCoordinate(0, 1 - idx)));
    }

    for (size_t idx = 0; idx < deviceTensors.size(); idx++) {
      std::cout << "Reorganizing tensor from Device #" << idx << std::endl;
      auto &deviceTensor = deviceTensors[idx];
      std::cout << " - Pulling tensor from Device #" << idx << std::endl;
      auto hostTensor = ttnn::from_device(deviceTensor, true, ttnn::DefaultQueueId);
      auto &submesh = targetSubmeshes[idx];

      std::cout << " - Pushing tensor to Device #" << (1 - idx) << std::endl;
      reorgTensors[1 - idx] = ttnn::to_device(hostTensor, submesh.get(), ttnn::DRAM_MEMORY_CONFIG, ttnn::DefaultQueueId);
    }
  }
  std::cout << "Aggregating single device tensors as a sharded multi device Tensor" << std::endl;
  ttnn::Tensor shardedTensor = ttnn::distributed::aggregate_as_tensor(reorgTensors, shardedInput.distributed_tensor_config());

  return shardedTensor;
}


ttnn::Tensor testingFunction3(ttnn::Tensor shardedInput, std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {

  ttnn::Tensor tensor0 = ttnn::cos(shardedInput);

  std::cout << "Retrieving single device tensors from sharded multi device tensor" << std::endl;
  std::vector<ttnn::Tensor> deviceTensors = ttnn::distributed::get_device_tensors(tensor0);
  std::vector<std::shared_ptr<ttnn::distributed::MeshDevice>> targetSubmeshes;
  for (size_t idx = 0; idx < deviceTensors.size(); idx++) {
    std::cout << " - Creating unit submesh of Device #" << (1 - idx) << std::endl;
    targetSubmeshes.push_back(meshDevice->create_submesh(ttnn::MeshShape(1, 1), ttnn::MeshCoordinate(0, 1 - idx)));
  }

  std::vector<ttnn::Tensor> reorgTensors(deviceTensors.size());

  for (size_t idx = 0; idx < deviceTensors.size(); idx++) {
    std::cout << "Reorganizing tensor from Device #" << idx << std::endl;
    auto &deviceTensor = deviceTensors[idx];
    std::cout << " - Pulling tensor from Device #" << idx << std::endl;
    auto hostTensor = ttnn::from_device(deviceTensor, true, ttnn::DefaultQueueId);
    auto &submesh = targetSubmeshes[idx];
    std::cout << " - Pushing tensor to Device #" << (1 - idx) << std::endl;
    reorgTensors[1 - idx] = ttnn::to_device(hostTensor, submesh.get(), ttnn::DRAM_MEMORY_CONFIG, ttnn::DefaultQueueId);
  }
  std::cout << "Aggregating single device tensors as a sharded multi device Tensor" << std::endl;
  ttnn::Tensor shardedTensor = ttnn::distributed::aggregate_as_tensor(reorgTensors, shardedInput.distributed_tensor_config());

  return shardedTensor;
}


ttnn::Tensor testingFunction4(ttnn::Tensor tensor, std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {
  ttnn::Tensor tensor0 = ttnn::cos(tensor);
  std::vector<ttnn::Tensor> reorgTensors(meshDevice->num_devices());
  std::vector<ttnn::Tensor> deviceTensors = ttnn::distributed::get_device_tensors(tensor0);
  for (size_t i = 0; i < deviceTensors.size(); i++)
  {
    auto hostTensor = ttnn::from_device(deviceTensors[i]);
    auto targetDevice = meshDevice->create_submesh(ttnn::MeshShape(1, 1), ttnn::MeshCoordinate(0, deviceTensors.size() - i - 1));
    reorgTensors[targetDevice->build_id()] = ttnn::to_device(hostTensor, targetDevice.get(), std::nullopt);
  }
  ttnn::Tensor shardedTensor = ttnn::distributed::aggregate_as_tensor(reorgTensors, tensor.distributed_tensor_config());
  return shardedTensor;
}

ttnn::Tensor testingFunction5(ttnn::Tensor tensor, std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {
  // ttnn::Tensor tensor0 = ttnn::cos(tensor);
  std::vector<ttnn::Tensor> deviceTensors = ttnn::distributed::get_device_tensors(tensor);
  std::vector<ttnn::Tensor> reorgTensors(meshDevice->num_devices());

  std::vector<std::shared_ptr<ttnn::distributed::MeshDevice>> targetSubmeshes;
  for (size_t i = 0; i < deviceTensors.size(); i++)
  {
    auto hostTensor = ttnn::from_device(deviceTensors[i]);
    auto targetDevice = meshDevice->create_submesh(ttnn::MeshShape(1, 1), ttnn::MeshCoordinate(0, deviceTensors.size() - i - 1));
    targetSubmeshes.push_back(targetDevice);  // store submesh to avoid deconstruction
    reorgTensors[targetDevice->build_id()] = ttnn::to_device(hostTensor, meshDevice.get(), std::nullopt);
  }
  targetSubmeshes.clear();

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

void testingCase1(std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {
  ttnn::Tensor inputTensor = create_inputs_for_testing();
  ttnn::Tensor shardedInput = shardTensor(inputTensor, meshDevice);
  ttnn::Tensor shardedOutput = testingFunction1(shardedInput, meshDevice);
  // Hangs on second from_device(..) API call.
  ttnn::Tensor outputTensor = unshardTensor(shardedOutput, meshDevice);
}
void testingCase2(std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {
  ttnn::Tensor inputTensor = create_inputs_for_testing();
  ttnn::Tensor shardedInput = shardTensor(inputTensor, meshDevice);
  ttnn::Tensor shardedOutput = testingFunction2(shardedInput, meshDevice);
  // Event ID is expected to increase. Wrapping not supported for sync. Completed event 1 but last recorded completed event is 2
  ttnn::Tensor outputTensor = unshardTensor(shardedOutput, meshDevice);
}
void testingCase3(std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {
  ttnn::Tensor inputTensor = create_inputs_for_testing();
  ttnn::Tensor shardedInput = shardTensor(inputTensor, meshDevice);
  ttnn::Tensor shardedOutput = testingFunction3(shardedInput, meshDevice);
  // Error aggregating multichip tensors: tensor shards must be allocated on the same mesh buffer. Consider moving tensors to host, aggregating, and re-uploading on device storage.
  ttnn::Tensor outputTensor = unshardTensor(shardedOutput, meshDevice);
}
void testingCase4(std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {
  ttnn::Tensor inputTensor = create_inputs_for_testing();
  ttnn::Tensor shardedInput = shardTensor(inputTensor, meshDevice);
  ttnn::Tensor shardedOutput = testingFunction4(shardedInput, meshDevice);
  // Error aggregating multichip tensors: tensor shards must be allocated on the same mesh buffer. Consider moving tensors to host, aggregating, and re-uploading on device storage.
  ttnn::Tensor outputTensor = unshardTensor(shardedOutput, meshDevice);
}
void testingCase5(std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice) {
  ttnn::Tensor inputTensor = create_inputs_for_testing();
  ttnn::Tensor shardedInput = shardTensor(inputTensor, meshDevice);
  ttnn::Tensor shardedOutput = testingFunction5(shardedInput, meshDevice);
  // Error aggregating multichip tensors: tensor shards must be allocated on the same mesh buffer. Consider moving tensors to host, aggregating, and re-uploading on device storage.
  ttnn::Tensor outputTensor = unshardTensor(shardedOutput, meshDevice);
}

int32_t main() {
  std::shared_ptr<ttnn::distributed::MeshDevice> meshDevice = openMeshDevice();
  testingCase5(meshDevice);
  // testingCase3(meshDevice);
  // testingCase1(meshDevice);
  return 0;
}
