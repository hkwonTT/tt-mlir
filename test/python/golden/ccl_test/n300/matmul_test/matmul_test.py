# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: SYSTEM_DESC_PATH=%system_desc_path% %python %s

import inspect
import torch

from ttmlir.test_utils import compile_to_flatbuffer, set_output_path
from ttmlir.ttir_builder import Operand, TTIRBuilder


@compile_to_flatbuffer(
    [
        (32, 128, 256),  # output47 output_47
        (32, 256, 128),  # output79 output_79
    ],
    targets=["ttnn"],
    module_dump=True,
)
def single_matmul(
    output47: Operand,
    output79: Operand,
    builder: TTIRBuilder,
):
    # output47_from_pt = torch.load("output_47.pt")
    # output79_from_pt = torch.load("output_79.pt")
    # golden = output47_from_pt @ output79_from_pt
    # builder.set_graph_input_output([output47_from_pt, output79_from_pt], [golden])

    output_81 = builder.matmul(output47, output79)
    return output_81


@compile_to_flatbuffer(
    [
        (32, 128, 256),  # output47 output_47
        (32, 256, 128),  # output79 output_79
    ],
    targets=["ttnn"],
    module_dump=True,
)
def multi_matmul_simul(
    output47: Operand,
    output79: Operand,
    builder: TTIRBuilder,
):
    output47_from_pt = torch.load("output_47.pt")
    output79_from_pt = torch.load("output_79.pt")
    output47_sharded = torch.chunk(output47_from_pt, 2, dim=2)
    output79_sharded = torch.chunk(output79_from_pt, 2, dim=1)
    partial_a = output47_sharded[0] @ output79_sharded[0]
    partial_b = output47_sharded[1] @ output79_sharded[1]
    golden = partial_a + partial_b
    builder.set_graph_input_output([output47_from_pt, output79_from_pt], [golden])

    output_81 = builder.matmul(output47, output79)
    return output_81


@compile_to_flatbuffer(
    [
        (32, 128, 256),  # arg0 output_47
        (32, 256, 128),  # arg1 output_79
    ],
    targets=["ttnn"],
    module_dump=True,
    mesh_shape=[1, 2],
)
def multi_matmul_partial(
    output47: Operand,
    output79: Operand,
    builder: TTIRBuilder,
):
    # output47_from_pt = torch.load("output_47.pt")
    # output79_from_pt = torch.load("output_79.pt")
    output47_from_pt = builder._get_golden_tensor(output47)
    output79_from_pt = builder._get_golden_tensor(output79)
    output47_sharded = torch.chunk(output47_from_pt, 2, dim=2)
    output79_sharded = torch.chunk(output79_from_pt, 2, dim=1)
    partial_a = output47_sharded[0] @ output79_sharded[0]
    partial_b = output47_sharded[1] @ output79_sharded[1]
    golden = partial_a + partial_b
    builder.set_graph_input_output([output47_from_pt, output79_from_pt], [golden])

    sharded_47 = builder.mesh_shard(
        output47,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 2],
        shard_dims=[-1, 2],
    )
    sharded_79 = builder.mesh_shard(
        output79,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 2, 1],
        shard_dims=[-1, 1],
    )
    partial = builder.matmul(sharded_47, sharded_79)
    reduced = builder.all_reduce(
        partial,
        reduce_type="#tt.reduce_type<sum>",
        cluster_axis=1,
    )
    output = builder.mesh_shard(
        reduced,
        shard_direction="#tt.shard_direction<shard_to_full>",
        shard_type="#tt.shard_type<replicate>",
        shard_shape=[1],
        shard_dims=[-1],
    )
    return output


if __name__ == "__main__":
    single_matmul()
    # multi_matmul_simul()
    # multi_matmul_partial()
