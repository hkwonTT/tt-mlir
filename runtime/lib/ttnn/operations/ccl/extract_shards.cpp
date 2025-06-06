// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/extract_shards.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::ExtractShardsOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in());

  std::vector<::ttnn::Tensor> deviceTensors =
      ::ttnn::distributed::get_device_tensors(inputTensor);

  for (size_t idx = 0; idx < deviceTensors.size(); idx++) {
    tensorPool.insertTTNNTensorAndValidate((*op->out())[idx],
                                           deviceTensors[idx]);
  }
}
} // namespace tt::runtime::ttnn::operations::ccl
