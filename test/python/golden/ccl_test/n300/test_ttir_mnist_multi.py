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
        (1, 1, 128, 1024),
        (1, 1, 1024, 256),
        (1, 1, 128, 256),
        (1, 1, 256, 64),
        (1, 1, 128, 64),
    ],
    module_dump=True,
    targets=["ttnn"],
)
def test_mnist(
    in0: Operand,  # Input 28x28 image
    in1: Operand,  # Weight 1
    in2: Operand,  # Bias 1
    in3: Operand,  # Weight 2
    in4: Operand,  # Bias 2
    builder: TTIRBuilder,
):
    global graph_input_tensors
    graph_input_tensors = []
    graph_input_tensors.append(builder._get_golden_tensor(in0))
    graph_input_tensors.append(builder._get_golden_tensor(in1))
    graph_input_tensors.append(builder._get_golden_tensor(in2))
    graph_input_tensors.append(builder._get_golden_tensor(in3))
    graph_input_tensors.append(builder._get_golden_tensor(in4))
    matmul_1 = builder.matmul(in0, in1)
    add_2 = builder.add(matmul_1, in2)
    relu_3 = builder.relu(add_2)
    matmul_5 = builder.matmul(relu_3, in3)
    add_6 = builder.add(matmul_5, in4)
    output = add_6
    # output = builder.softmax(add_6, dimension=3)
    global graph_output_tensors
    global_output_tensors = []
    graph_output_tensors.append(builder._get_golden_tensor(output))
    return output


@compile_to_flatbuffer(
    [
        (1, 1, 128, 1024),
        (1, 1, 1024, 256),
        (1, 1, 128, 256),
        (1, 1, 256, 64),
        (1, 1, 128, 64),
    ],
    targets=["ttnn"],
    mesh_shape=[1, 2],
    module_dump=True,
)
def test_mnist_multidevice(
    in0: Operand,  # Input 28x28 image
    in1: Operand,  # Weight 1
    in2: Operand,  # Bias 1
    in3: Operand,  # Weight 2
    in4: Operand,  # Bias 2
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
    partial_matmul_1 = builder.matmul(sharded_in0, sharded_in1)
    matmul_1 = builder.reduce_scatter(
        partial_matmul_1,
        reduce_type="#tt.reduce_type<sum>",
        scatter_dim=3,
        cluster_axis=1,
    )
    sharded_in2 = builder.mesh_shard(
        in2,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 1, 2],
        shard_dims=[-1, 3],
    )
    add_2 = builder.add(matmul_1, sharded_in2)
    relu_3 = builder.relu(add_2)

    sharded_in3 = builder.mesh_shard(
        in3,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 2, 1],
        shard_dims=[-1, 2],
    )
    partial_matmul_5 = builder.matmul(relu_3, sharded_in3)
    matmul_5 = builder.reduce_scatter(
        partial_matmul_5,
        reduce_type="#tt.reduce_type<sum>",
        scatter_dim=3,
        cluster_axis=1,
    )
    sharded_in4 = builder.mesh_shard(
        in4,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 1, 2],
        shard_dims=[-1, 3],
    )
    add_6 = builder.add(matmul_5, sharded_in4)

    output = builder.mesh_shard(
        add_6,
        shard_direction="#tt.shard_direction<shard_to_full>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=[1, 1, 1, 2],
        shard_dims=[-1, 3],
    )
    # output = builder.softmax(add_6, dimension=3)
    return output


if __name__ == "__main__":
    test_mnist()
    test_mnist_multidevice()
