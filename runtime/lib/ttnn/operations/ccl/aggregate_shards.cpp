// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/aggregate_shards.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::AggregateShardsOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  std::vector<::ttnn::Tensor> inputs;
  for (const auto &input : *op->inputs()) {
    const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(input);
    inputs.push_back(in);
  }
  auto config = tt::runtime::ttnn::operations::utils::
      distributedTensorConfigFromFlatbuffer(op->config());

  ::ttnn::Tensor out = ::ttnn::distributed::aggregate_as_tensor(inputs, config);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
