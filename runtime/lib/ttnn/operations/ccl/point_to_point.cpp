// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/point_to_point.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include <optional>

/*
This is a temporary host fallback to ttnn::PointToPoint(..) API.
*/
namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::PointToPointOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in());
  DEBUG_ASSERT(!::tt::runtime::ttnn::utils::inSystemMemory(op->in()),
               "Calling ttnn::from_device on a host tensor");
  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();

  ::ttnn::MeshCoordinate coord(0, 0);
  auto targetSubmesh =
      meshDevice.create_submesh(::ttnn::MeshShape(1, 1), coord);
  auto id = targetSubmesh->get_device(::ttnn::MeshCoordinate(0, 0))->id();
  LOG_DEBUG("target device ID :", id, " (", coord.coords()[0], ",",
            coord.coords()[1], ")");

  ::ttnn::Tensor hostTensor = ::ttnn::from_device(inputTensor);
  ::ttnn::Tensor deviceTensor =
      ::ttnn::to_device(hostTensor, targetSubmesh.get(), std::nullopt);

  // ::ttnn::Tensor deviceTensor = inputTensor;

  /*
  ::ttnn::IDevice *targetDevice = meshDevice.get_device(coord);

  ::ttnn::Tensor hostTensor = ::ttnn::from_device(inputTensor);
  bool isMultiDeviceTensor =
      (hostTensor.storage_type() == ::ttnn::StorageType::MULTI_DEVICE_HOST);
  if (isMultiDeviceTensor) {
    hostTensor =
        ::ttnn::Tensor(tt_metal::host_buffer::get_host_buffer(hostTensor),
                       hostTensor.tensor_spec());
  }
  ::ttnn::Tensor deviceTensor =
      ::ttnn::to_device(hostTensor, targetDevice, inputTensor.memory_config());
  if (isMultiDeviceTensor) {
    ::ttnn::Tensor temp = ::ttnn::Tensor(deviceTensor.storage(),
  deviceTensor.tensor_spec(), inputTensor.distributed_tensor_config());
    deviceTensor = temp;
  }
  */

  tensorPool.insertTTNNTensorAndValidate(op->out(), deviceTensor);
}
} // namespace tt::runtime::ttnn::operations::ccl
