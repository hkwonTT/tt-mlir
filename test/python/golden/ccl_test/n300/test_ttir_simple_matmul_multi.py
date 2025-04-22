# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: SYSTEM_DESC_PATH=%system_desc_path% %python %s

import inspect
import torch
import numpy as np

from ttmlir.test_utils import compile_to_flatbuffer, set_output_path
from ttmlir.ttir_builder import Operand, TTIRBuilder
from ttmlir.passes import GoldenTensor, DataType

graph_input_tensors = []
graph_output_tensors = []


@compile_to_flatbuffer(
    [
        (64, 1024),
        (1024, 256),
        (64, 256),
    ],
    module_dump=True,
    targets=["ttnn"],
)
def test_matmul(
    in0: Operand,  # Input 28x28 image
    in1: Operand,  # Weight 1
    in2: Operand,  # Bias 1
    builder: TTIRBuilder,
):
    global graph_input_tensors
    graph_input_tensors = []
    graph_input_tensors.append(builder._get_golden_tensor(in0))
    graph_input_tensors.append(builder._get_golden_tensor(in1))
    graph_input_tensors.append(builder._get_golden_tensor(in2))
    matmul_1 = builder.matmul(in0, in1)
    add_2 = builder.add(matmul_1, in2)
    output = add_2
    global graph_output_tensors
    global_output_tensors = []
    graph_output_tensors.append(builder._get_golden_tensor(output))
    return output


@compile_to_flatbuffer(
    [
        (64, 1024),
        (1024, 256),
        (64, 256),
    ],
    targets=["ttnn"],
    mesh_shape=[1, 2],
    module_dump=True,
)
def test_matmul_multidevice(
    in0: Operand,  # Input 28x28 image
    in1: Operand,  # Weight 1
    in2: Operand,  # Bias 1
    builder: TTIRBuilder,
):
    global graph_input_tensors
    global graph_output_tensors
    builder.set_graph_input_output(graph_input_tensors, graph_output_tensors)

    sharded_in0 = builder.mesh_shard(
        in0,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 2],
        shard_dims=[-1, 1],
    )
    sharded_in1 = builder.mesh_shard(
        in1,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[2, 1],
        shard_dims=[-1, 0],
    )
    sharded_matmul_1 = builder.matmul(sharded_in0, sharded_in1)
    reduced_matmul_1 = builder.all_reduce(
        sharded_matmul_1,
        reduce_type="#tt.reduce_type<sum>",
        cluster_axis=1,
    )
    matmul_1 = builder.mesh_shard(
        reduced_matmul_1,
        shard_direction="#tt.shard_direction<shard_to_full>",
        shard_type="#tt.shard_type<replicate>",
        shard_shape=[1],
        shard_dims=[-1],
    )
    sharded_2_matmul_1 = builder.mesh_shard(
        matmul_1,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 2],
        shard_dims=[-1, 1],
    )
    sharded_in2 = builder.mesh_shard(
        in2,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 2],
        shard_dims=[-1, 1],
    )
    sharded_add_2 = builder.add(sharded_2_matmul_1, sharded_in2)
    add_2 = builder.mesh_shard(
        sharded_add_2,
        shard_direction="#tt.shard_direction<shard_to_full>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 2],
        shard_dims=[-1, 1],
    )
    # add_2 = builder.add(matmul_1, in2)
    output = add_2
    return output


@compile_to_flatbuffer(
    [
        (1, 1, 64, 1024),
        (1, 1, 1024, 256),
        (1, 1, 64, 256),
    ],
    module_dump=True,
    targets=["ttnn"],
)
def test_matmul_ver2(
    in0: Operand,  # Input 28x28 image
    in1: Operand,  # Weight 1
    in2: Operand,  # Bias 1
    builder: TTIRBuilder,
):
    global graph_input_tensors
    graph_input_tensors = []
    graph_input_tensors.append(builder._get_golden_tensor(in0))
    graph_input_tensors.append(builder._get_golden_tensor(in1))
    graph_input_tensors.append(builder._get_golden_tensor(in2))
    matmul_1 = builder.matmul(in0, in1)
    add_2 = builder.add(matmul_1, in2)
    output = add_2
    global graph_output_tensors
    global_output_tensors = []
    graph_output_tensors.append(builder._get_golden_tensor(output))
    return output


@compile_to_flatbuffer(
    [
        (1, 1, 64, 1024),
        (1, 1, 1024, 256),
        (1, 1, 64, 256),
    ],
    targets=["ttnn"],
    mesh_shape=[1, 2],
    module_dump=True,
)
def test_matmul_multidevice_ver2(
    in0: Operand,  # Input 28x28 image
    in1: Operand,  # Weight 1
    in2: Operand,  # Bias 1
    builder: TTIRBuilder,
):
    global graph_input_tensors
    global graph_output_tensors
    builder.set_graph_input_output(graph_input_tensors, graph_output_tensors)

    sharded_in0 = builder.mesh_shard(
        in0,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 1, 2],
        shard_dims=[-1, 3],
    )
    sharded_in1 = builder.mesh_shard(
        in1,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 2, 1],
        shard_dims=[-1, 2],
    )
    sharded_matmul_1 = builder.matmul(sharded_in0, sharded_in1)
    reduced_matmul_1 = builder.reduce_scatter(
        sharded_matmul_1,
        reduce_type="#tt.reduce_type<sum>",
        scatter_dim=3,
        cluster_axis=1,
    )
    # matmul_1 = builder.mesh_shard(
    #     reduced_matmul_1,
    #     shard_direction="#tt.shard_direction<shard_to_full>",
    #     shard_type="#tt.shard_type<devices>",
    #     shard_shape=[1, 1, 1, 2],
    #     shard_dims=[-1, 3],
    # )
    # sharded_2_matmul_1 = builder.mesh_shard(
    #     matmul_1,
    #     shard_direction="#tt.shard_direction<full_to_shard>",
    #     shard_type="#tt.shard_type<devices>",
    #     shard_shape=[1, 1, 1, 2],
    #     shard_dims=[-1, 3],
    # )
    sharded_in2 = builder.mesh_shard(
        in2,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 1, 2],
        shard_dims=[-1, 3],
    )
    sharded_add_2 = builder.add(reduced_matmul_1, sharded_in2)
    add_2 = builder.mesh_shard(
        sharded_add_2,
        shard_direction="#tt.shard_direction<shard_to_full>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 1, 2],
        shard_dims=[-1, 3],
    )
    output = add_2
    return output


if __name__ == "__main__":
    test_matmul()
    test_matmul_multidevice()
    # test_matmul_ver2()
    # test_matmul_multidevice_ver2()
