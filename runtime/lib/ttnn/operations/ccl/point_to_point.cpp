// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/point_to_point.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
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

  // ToDo: Implement something

  tensorPool.insertTTNNTensorAndValidate(op->out(), inputTensor);
}
} // namespace tt::runtime::ttnn::operations::ccl
