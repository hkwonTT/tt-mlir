// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/point_to_point.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include <optional>
#include <ttnn/distributed/types.hpp>

/*
This is a temporary host fallback to ttnn::PointToPoint(..) API.
*/
namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::PointToPointOp *op,
         ProgramContext &context) {
  DEBUG_ASSERT(!::tt::runtime::ttnn::utils::inSystemMemory(op->in()),
               "Calling ttnn::from_device on a host tensor");
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in());
  ::ttnn::MeshCoordinate targetCoord(op->dest_coord()->y(),
                                     op->dest_coord()->x());
  auto targetId =
      context.getMeshDevice().get_view().find_device_id(targetCoord);
  auto targetMeshDevice = context.getUnitMeshDevice(targetId);

  LOG_DEBUG("target device ID :", targetId, " (", targetCoord.coords()[0], ",",
            targetCoord.coords()[1], ")");

  ::ttnn::Tensor hostTensor = ::ttnn::from_device(inputTensor);
  ::ttnn::Tensor deviceTensor =
      ::ttnn::to_device(hostTensor, targetMeshDevice.get(), std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), deviceTensor);
}
} // namespace tt::runtime::ttnn::operations::ccl
