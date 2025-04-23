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

from helpers import *


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
    save_all_input_goldens(*locals().values(), builder=builder)
    output = matmul_1 = builder.matmul(in0, in1)
    output = add_2 = builder.add(matmul_1, in2)
    output = relu_3 = builder.relu(add_2)
    output = matmul_5 = builder.matmul(relu_3, in3)
    output = add_6 = builder.add(matmul_5, in4)
    # output = builder.softmax(add_6, dimension=3)

    save_all_output_goldens(output, builder=builder)
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
    in0_sharded = full_to_shard_device(in0, builder, 3)
    in1_sharded = full_to_shard_device(in1, builder, 2)
    partial_matmul_1 = builder.matmul(in0_sharded, in1_sharded)
    matmul_1 = builder.reduce_scatter(
        partial_matmul_1,
        reduce_type="#tt.reduce_type<sum>",
        scatter_dim=3,
        cluster_axis=1,
    )
    in2_sharded = full_to_shard_device(in2, builder, 3)
    add_2 = builder.add(matmul_1, in2_sharded)
    relu_3 = builder.relu(add_2)
    in3_sharded = full_to_shard_device(in3, builder, 2)
    partial_matmul_5 = builder.matmul(relu_3, in3_sharded)
    matmul_5 = builder.reduce_scatter(
        partial_matmul_5,
        reduce_type="#tt.reduce_type<sum>",
        scatter_dim=3,
        cluster_axis=1,
    )
    in4_sharded = full_to_shard_device(in4, builder, 3)
    add_6 = builder.add(matmul_5, in4_sharded)
    output = shard_to_full_device(add_6, builder, 3)
    # output = builder.softmax(add_6, dimension=3)
    set_graph_goldens(builder)
    return output


if __name__ == "__main__":
    test_mnist()
    test_mnist_multidevice()
