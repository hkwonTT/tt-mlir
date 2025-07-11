# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from typing import List, Tuple
from ttir_builder.utils import compile_to_flatbuffer
from ttir_builder import Operand, TTIRBuilder, Shape
import test_util

pytestmark = pytest.mark.n300


def pseudo_golden_all_gather(
    input_tensor: torch.Tensor,
    mesh_shape,
    shard_dims,
    all_gather_dim: int,
    cluster_axis: int,
):
    shards = test_util.shardTensor2dMesh(input_tensor, mesh_shape, shard_dims)
    if mesh_shape[cluster_axis] == 1:
        return shards[0]
    output_tensor = torch.cat(shards, dim=all_gather_dim)
    return output_tensor


def test_all_gather(
    shape: Shape,
    mesh_shape,
    shard_dims,
    shard_shape,
    cluster_axis,
    replica_groups,
    all_gather_dim,
    request,
):
    if not test_util.shape_divisible(shape, shard_shape):
        pytest.skip(f"shape {shape} is not divisible by shard_shape {shard_shape}")
    print(
        f"shape={shape}, mesh_shape={mesh_shape}, shard_dims={shard_dims}, shard_shape={shard_shape}, cluster_axis={cluster_axis}, replica_groups={replica_groups}"
    )

    def all_gather(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = pseudo_golden_all_gather(
            input, mesh_shape, shard_dims, all_gather_dim
        )
        builder.set_graph_input_output([input], [golden_output])

        sharded = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape,
            shard_dims=shard_dims,
        )
        gathered = builder.all_gather(
            sharded,
            all_gather_dim=all_gather_dim,
            cluster_axis=cluster_axis,
        )
        return builder.mesh_shard(
            gathered,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<replicate>",
            shard_shape=(1,),
            shard_dims=(-1,),
        )

    def seq2str(seq, delim="x"):
        return delim.join(str(num) for num in seq)

    def generate_test_base():
        return f"test-all-gather_input_{seq2str(shape)}_mesh_{seq2str(mesh_shape)}_all-gather-dim_{all_gather_dim}_cluster_{cluster_axis}_shard-dims_{seq2str(shard_dims)}_shard-shape_{seq2str(shard_shape)}"

    compile_to_flatbuffer(
        all_gather,
        [shape],
        mesh_shape=mesh_shape,
        test_base=generate_test_base(),
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        module_dump=True,
    )
