// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/point_to_point.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/utils.h"

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

  ::ttnn::MeshCoordinate coord(op->dest_coord()->y(), op->dest_coord()->x());

  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
  ::ttnn::IDevice *targetDevice = meshDevice.get_device(coord);

  ::ttnn::Tensor hostTensor = ::ttnn::from_device(inputTensor);
  ::ttnn::Tensor out =
      ::ttnn::to_device(hostTensor, targetDevice, inputTensor.memory_config());

  tensorPool.insertTTNNTensorAndValidate(op->out(), inputTensor);
}
} // namespace tt::runtime::ttnn::operations::ccl
