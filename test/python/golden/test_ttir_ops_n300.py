# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from typing import List, Tuple
from ttmlir.test_utils import compile_to_flatbuffer
from ttmlir.ttir_builder import Operand, TTIRBuilder, Shape

pytestmark = pytest.mark.n300


def pseudo_golden_all_gather(
    input_tensor: torch.Tensor, shard_dim: int, gather_dim: int
):
    sharded = torch.chunk(input_tensor, 2, shard_dim)
    return torch.concat(sharded, gather_dim)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 128, 128),  # 0 PASS
        (1, 32, 120, 128),  # 1 PASS
        pytest.param((1, 32, 128, 120), marks=pytest.mark.fails_golden),  # 2 ERROR
        pytest.param((1, 32, 120, 120), marks=pytest.mark.fails_golden),  # 3 ERROR
        (1, 32, 60, 128),  # 4 PASS
        pytest.param((1, 32, 128, 60), marks=pytest.mark.fails_golden),  # 5 ERROR
        pytest.param((1, 32, 60, 60), marks=pytest.mark.fails_golden),  # 6 ERROR
        (1, 32, 30, 128),  # 7 PASS
        pytest.param((1, 32, 128, 30), marks=pytest.mark.fails_golden),  # 8 ERROR
        pytest.param((1, 32, 30, 30), marks=pytest.mark.fails_golden),  # 9 ERROR
        (1, 32, 2, 128),  # 10 PASS
        pytest.param((1, 32, 128, 2), marks=pytest.mark.fails_golden),  # 11 ERROR
        pytest.param((1, 32, 2, 2), marks=pytest.mark.fails_golden),  # 12 ERROR
        pytest.param((1, 1, 1, 2), marks=pytest.mark.fails_golden),  # 13 ERROR
        pytest.param((1, 1, 10, 10), marks=pytest.mark.fails_golden),  # 14 ERROR
    ],
)
@pytest.mark.parametrize(
    "shard_dim",
    [2, 3],
)
@pytest.mark.parametrize(
    "gather_dim",
    [2, 3],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_all_gather(
    shape: Shape, mesh_shape: Tuple[int, int], shard_dim: int, gather_dim: int, request
):
    if shape[shard_dim] % mesh_shape[1] != 0:
        pytest.skip("not divisible")

    def gen_test_name():
        return f"all-gather_{'_'.join(map(str, shape))}_{shard_dim}_{gather_dim}"

    def all_gather(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = pseudo_golden_all_gather(input, shard_dim, gather_dim)
        builder.set_graph_input_output([input], [golden_output])
        shard_shape = [1] * len(shape)
        shard_shape[shard_dim] = mesh_shape[1]
        shard_dims = (-1, shard_dim)

        sharded = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=tuple(shard_shape),
            shard_dims=shard_dims,
        )
        gathered = builder.all_gather(
            sharded,
            all_gather_dim=gather_dim,
            cluster_axis=1,
        )
        return builder.mesh_shard(
            gathered,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<replicate>",
            shard_shape=(1,),
            shard_dims=(-1,),
        )

    compile_to_flatbuffer(
        all_gather, [shape], mesh_shape=mesh_shape, test_base=gen_test_name()
    )


@pytest.mark.parametrize("dim0", [0, 1])
@pytest.mark.parametrize("dim1", [0, 1, 32])
@pytest.mark.parametrize(
    "dim2", [1, 2, 16, 32, 62, 63, 64, 65, 66, 126, 127, 128, 129, 130]
)
@pytest.mark.parametrize(
    "dim3", [1, 2, 16, 32, 62, 63, 64, 65, 66, 126, 127, 128, 129, 130]
)
@pytest.mark.parametrize(
    "shard_dim",
    [0, 1, 2, 3],
)
@pytest.mark.parametrize(
    "gather_dim",
    [0, 1, 2, 3],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_combi_all_gather(
    dim0: int,
    dim1: int,
    dim2: int,
    dim3: int,
    mesh_shape: Tuple[int, int],
    shard_dim: int,
    gather_dim: int,
    request,
):
    shape = []
    if dim0 != 0:
        shape.append(dim0)
    if dim1 != 0:
        shape.append(dim1)
    if dim2 != 0:
        shape.append(dim2)
    if dim3 != 0:
        shape.append(dim3)
    shape = tuple(shape)
    if shape[shard_dim] % mesh_shape[1] != 0:
        pytest.skip("not divisible")
    if shard_dim > (len(shape) - 1):
        pytest.skip("over dim")
    if gather_dim > (len(shape) - 1):
        pytest.skip("over dim")

    def all_gather(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = pseudo_golden_all_gather(input, shard_dim, gather_dim)
        builder.set_graph_input_output([input], [golden_output])
        shard_shape = [1] * len(shape)
        shard_shape[shard_dim] = mesh_shape[1]
        shard_dims = (-1, shard_dim)

        sharded = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=tuple(shard_shape),
            shard_dims=shard_dims,
        )
        gathered = builder.all_gather(
            sharded,
            all_gather_dim=gather_dim,
            cluster_axis=1,
        )
        return builder.mesh_shard(
            gathered,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<replicate>",
            shard_shape=(1,),
            shard_dims=(-1,),
        )

    compile_to_flatbuffer(
        all_gather, [shape], mesh_shape=mesh_shape, test_base=request.node.name
    )


def pseudo_golden_mesh_shard_replicate(
    input_tensor: torch.Tensor,
):
    output_tensor = torch.chunk(input_tensor, 2, -1)
    return output_tensor[0]


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 128, 128),
        (1, 32, 120, 128),
        (1, 32, 128, 120),
        (1, 32, 120, 120),
        (1, 32, 60, 128),
        (1, 32, 128, 60),
        (1, 32, 60, 60),
        (1, 32, 30, 128),
        (1, 32, 128, 30),
        (1, 32, 30, 30),
        (1, 32, 2, 128),
        (1, 32, 128, 2),
        (1, 32, 2, 2),
        (1, 1, 1, 2),
        (1, 1, 10, 10),
    ],
)
@pytest.mark.parametrize(
    "shard_dim",
    [2],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_mesh_shard_replicate(
    shape: Shape, mesh_shape: Tuple[int, int], shard_dim: int, request
):
    def all_gather(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = pseudo_golden_mesh_shard_replicate(input)
        builder.set_graph_input_output([input], [golden_output])
        shard_shape = [1] * len(shape)
        shard_shape[shard_dim] = mesh_shape[1]
        shard_dims = (-1, shard_dim)
        print(f"shard_shape = {shard_shape}")
        print(f"shard_dims = {shard_dims}")
        if shape[shard_dim] % mesh_shape[1] != 0:
            pytest.skip("not divisible")

        sharded = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=tuple(shard_shape),
            shard_dims=shard_dims,
        )
        return builder.mesh_shard(
            sharded,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<replicate>",
            shard_shape=(1,),
            shard_dims=(-1,),
        )

    compile_to_flatbuffer(
        all_gather, [shape], mesh_shape=mesh_shape, test_base=request.node.name
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 128, 128),
        (1, 32, 120, 128),
        (1, 32, 128, 120),
        (1, 32, 120, 120),
        (1, 32, 60, 128),
        (1, 32, 128, 60),
        (1, 32, 60, 60),
        (1, 32, 30, 128),
        (1, 32, 128, 30),
        (1, 32, 30, 30),
        (1, 32, 2, 128),
        (1, 32, 128, 2),
        (1, 32, 2, 2),
        (1, 1, 1, 2),
        (1, 1, 10, 10),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_mesh_shard_devices(shape: Shape, mesh_shape: Tuple[int, int], request):
    def all_gather(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = pseudo_golden_all_gather(input)
        builder.set_graph_input_output([input], [golden_output])

        sharded = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )
        return builder.mesh_shard(
            sharded,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )

    compile_to_flatbuffer(
        all_gather, [shape], mesh_shape=mesh_shape, test_base=request.node.name
    )


def pseudo_golden_all_reduce(input_tensor: torch.Tensor):
    shard_1, shard_2 = torch.chunk(input_tensor, 2, dim=3)
    output_tensor = shard_1 + shard_2
    return output_tensor


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 128, 512),
        (1, 1, 128, 132),
        (1, 1, 126, 128),
        (1, 1, 1, 128),
        (1, 1, 1, 64),
        (1, 1, 1, 124),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_all_reduce(shape: Shape, mesh_shape: Tuple[int, int], request):
    def all_reduce(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = pseudo_golden_all_reduce(input)
        builder.set_graph_input_output([input], [golden_output])

        sharded = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )
        reduced = builder.all_reduce(
            sharded,
            reduce_type="#tt.reduce_type<sum>",
            cluster_axis=1,
        )
        return builder.mesh_shard(
            reduced,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<replicate>",
            shard_shape=(1,),
            shard_dims=(-1,),
        )

    compile_to_flatbuffer(
        all_reduce, [shape], mesh_shape=mesh_shape, test_base=request.node.name
    )


def pseudo_golden_reduce_scatter(
    input_tensor: torch.Tensor,
    scatter_dim: int,
):
    shard_1, shard_2 = torch.chunk(input_tensor, 2, dim=scatter_dim)
    output_tensor = shard_1 + shard_2
    return output_tensor


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 128, 128),
        (1, 1, 128, 124),
        (1, 1, 124, 128),
        (1, 1, 124, 124),
        (1, 1, 32, 64),
        (1, 1, 32, 60),
        (1, 1, 30, 64),
        (1, 1, 30, 60),
        (1, 1, 1, 64),
        (1, 1, 1, 60),
        (1, 1, 1, 4),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_reduce_scatter(shape: Shape, mesh_shape: Tuple[int, int], request):
    def reduce_scatter(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = pseudo_golden_reduce_scatter(input, 3)
        builder.set_graph_input_output([input], [golden_output])

        sharded = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )
        reduced = builder.reduce_scatter(
            sharded,
            reduce_type="#tt.reduce_type<sum>",
            scatter_dim=3,
            cluster_axis=1,
        )
        return builder.mesh_shard(
            reduced,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )

    compile_to_flatbuffer(
        reduce_scatter, [shape], mesh_shape=mesh_shape, test_base=request.node.name
    )


def pseudo_golden_collective_permute(
    input_tensor: torch.Tensor,
    source_target_pairs: List[Tuple[int, int]],
):
    shards = list(torch.chunk(input_tensor, 2, dim=3))
    permuted_tensor = shards.copy()
    for source, target in source_target_pairs:
        permuted_tensor[target] = shards[source]
    result_tensor = torch.cat(permuted_tensor, dim=3)
    return result_tensor


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 128, 128),
        (1, 1, 64, 64),
        (1, 1, 64, 62),
        (1, 1, 62, 64),
        (1, 1, 62, 62),
        (1, 1, 32, 32),
        (1, 1, 32, 34),
        (1, 1, 34, 32),
        (1, 1, 34, 34),
        (1, 1, 32, 16),
        (1, 1, 32, 14),
        (1, 1, 32, 18),
        (1, 1, 32, 2),
        (1, 1, 32, 4),
        (1, 1, 31, 32),
        (1, 1, 1, 32),
        (1, 1, 2, 32),
        (1, 1, 33, 32),
    ],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_collective_permute(shape: Shape, mesh_shape: Tuple[int, int], request):
    def collective_permute(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = pseudo_golden_collective_permute(input, [(0, 1), (1, 0)])
        builder.set_graph_input_output([input], [golden_output])

        sharded = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )
        reduced = builder.collective_permute(
            sharded,
            source_target_pairs=[(0, 1), (1, 0)],
        )
        return builder.mesh_shard(
            reduced,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )

    compile_to_flatbuffer(
        collective_permute, [shape], mesh_shape=mesh_shape, test_base=request.node.name
    )


@pytest.mark.parametrize("shapes", [[(2048, 196), (196, 4096)]])
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_matmul_1x2(shapes: List[Shape], mesh_shape: Tuple[int, int], request):
    def matmul_1x2(in0: Operand, in1: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        weight = builder._get_golden_tensor(in1)
        golden_output = torch.matmul(input, weight)
        builder.set_graph_input_output([input, weight], [golden_output])

        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 2),
            shard_dims=(-1, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(2, 1),
            shard_dims=(-1, 0),
        )
        partial_matmul = builder.matmul(sharded_in0, sharded_in1)
        reduced = builder.all_reduce(
            partial_matmul,
            reduce_type="#tt.reduce_type<sum>",
            cluster_axis=1,
        )
        return builder.mesh_shard(
            reduced,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<replicate>",
            shard_shape=(1,),
            shard_dims=(-1,),
        )

    compile_to_flatbuffer(
        matmul_1x2, shapes, mesh_shape=mesh_shape, test_base=request.node.name
    )


@pytest.mark.parametrize("shape", [(1, 256, 64, 256)])
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_neg_1x2_dim_3(shape: Shape, mesh_shape: Tuple[int, int], request):
    def neg_1x2_dim_3(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = torch.neg(input)
        builder.set_graph_input_output([input], [golden_output])

        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 1, 1, 2),
            shard_dims=(-1, 3),
        )

    compile_to_flatbuffer(
        neg_1x2_dim_3,
        [shape],
        mesh_shape=mesh_shape,
        test_base=request.node.name,
    )


@pytest.mark.parametrize("shape", [(1, 256, 64, 256)])
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_neg_1x2_dim_1(shape: Shape, mesh_shape: Tuple[int, int], request):
    def neg_1x2_dim_1(in0: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        golden_output = torch.neg(input)
        builder.set_graph_input_output([input], [golden_output])

        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 2, 1, 1),
            shard_dims=(-1, 1),
        )
        neg_output = builder.neg(sharded_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 2, 1, 1),
            shard_dims=(-1, 1),
        )

    compile_to_flatbuffer(
        neg_1x2_dim_1,
        [shape],
        mesh_shape=mesh_shape,
        test_base=request.node.name,
    )


@pytest.mark.parametrize("shapes", [[(512, 1024), (512, 1024)]])
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_eltwise_multidevice(shapes: List[Shape], mesh_shape: Tuple[int, int], request):
    def eltwise_multidevice(in0: Operand, in1: Operand, builder: TTIRBuilder):
        input = builder._get_golden_tensor(in0)
        weight = builder._get_golden_tensor(in1)
        golden_output = torch.add(input, weight)
        builder.set_graph_input_output([input, weight], [golden_output])

        sharded_in0 = builder.mesh_shard(
            in0,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 2),
            shard_dims=(-1, 1),
        )
        sharded_in1 = builder.mesh_shard(
            in1,
            shard_direction="#tt.shard_direction<full_to_shard>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 2),
            shard_dims=(-1, 1),
        )
        partial_sum = builder.add(sharded_in0, sharded_in1)
        return builder.mesh_shard(
            partial_sum,
            shard_direction="#tt.shard_direction<shard_to_full>",
            shard_type="#tt.shard_type<devices>",
            shard_shape=(1, 2),
            shard_dims=(-1, 1),
        )

    compile_to_flatbuffer(
        eltwise_multidevice, shapes, mesh_shape=mesh_shape, test_base=request.node.name
    )
