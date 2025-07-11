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


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 128, 128),
        (1, 32, 120, 128),
        (1, 32, 60, 128),
        (1, 32, 30, 128),
        (1, 32, 2, 128),
        (1, 32, 128, 120),
        (1, 32, 120, 120),
        (1, 32, 128, 60),
        (1, 32, 60, 60),
        (1, 32, 128, 30),
        (1, 32, 30, 30),
        (1, 32, 128, 2),
        (1, 32, 2, 2),
        (1, 1, 10, 10),
        (128, 32, 128, 1),
        (128, 32, 1, 1),
    ],
)
@pytest.mark.parametrize(
    "mesh_shape, shard_dims, shard_shape, cluster_axis, replica_groups",
    list(
        test_util.sweep_all_sharding_configs(
            input_rank=4, num_devices=2, min_mesh_dim=2, max_mesh_dim=2
        )
    ),
)
@pytest.mark.parametrize("all_gather_dim", [0, 1, 2, 3])
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
            input, mesh_shape, shard_dims, all_gather_dim, cluster_axis
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
