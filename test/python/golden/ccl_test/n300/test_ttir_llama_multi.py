# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: SYSTEM_DESC_PATH=%system_desc_path% %python %s

import inspect

from ttmlir.test_utils import compile_to_flatbuffer, set_output_path
from ttmlir.ttir_builder import Operand, TTIRBuilder

from helpers import *


@compile_to_flatbuffer(
    [
        (1, 128, 4096),  # arg0
        (1, 1, 128, 128),  # arg1
        (1, 128),  # arg2
        (1, 64, 1),  # arg3
        (1, 32, 64, 128),  # arg4
        (1, 1),  # arg5
        (1, 32, 64, 128),  # arg6
        (1, 32, 64, 128),  # arg7
        (1, 1),  # arg8
        (1, 32, 64, 128),  # arg9
        (1, 1),  # arg10
        (4096, 4096),  # arg11
        (4096, 4096),  # arg12
        # (4096, 4096),  # arg13
        # (4096, 4096),  # arg14
    ],
    targets=["ttnn"],
    module_dump=True,
)
def test_llama_attention(
    arg0: Operand,
    arg1: Operand,
    arg2: Operand,
    arg3: Operand,
    arg4: Operand,
    arg5: Operand,
    arg6: Operand,
    arg7: Operand,
    arg8: Operand,
    arg9: Operand,
    arg10: Operand,
    arg11: Operand,
    arg12: Operand,
    # arg13: Operand,
    # arg14: Operand,
    builder: TTIRBuilder,
):
    save_all_input_goldens(
        [v for k, v in locals().items() if k != "builder"], builder=builder
    )
    output1 = builder.squeeze(arg0, 0)  # [128, 4096]

    output3 = builder.matmul(output1, arg11)  # [128, 4096]
    output5 = builder.reshape(output3, (1, 128, 32, 128))  # [1, 128, 32, 128]
    output7 = builder.transpose(output5, -3, -2)  # [1, 32, 128, 128]
    output9 = builder.unsqueeze(arg2, 1)  # [1, 1, 128]
    output11 = builder.matmul(arg3, output9)  # [1, 64, 128]
    output13 = builder.transpose(output11, -2, -1)  # [1, 128, 64]
    output15 = builder.concat([output13, output13], -1)  # [1, 128, 128]
    output17 = builder.cos(output15)  # [1, 128, 128]
    output19 = builder.unsqueeze(output17, 1)  # [1, 1, 128, 128]
    output21 = builder.multiply(output7, output19)  # [1, 32, 128, 128]
    output23 = builder.transpose(output7, -2, -1)  # [1, 32, 128, 128]
    output25 = builder.matmul(arg4, output23)  # [1, 32, 64, 128]
    output27 = builder.transpose(output25, -2, -1)  # [1, 32, 128, 64]
    output29 = builder.multiply(output27, arg5)  # [1, 32, 128, 64]
    output31 = builder.transpose(output7, -2, -1)  # [1, 32, 128, 128]
    output33 = builder.matmul(arg6, output31)  # [1, 32, 64, 128]
    output35 = builder.transpose(output33, -2, -1)  # [1, 32, 128, 64]
    output37 = builder.concat([output29, output35], -1)  # [1, 32, 128, 128]
    output39 = builder.sin(output15)  # [1, 128, 128]
    output41 = builder.unsqueeze(output39, 1)  # [1, 1, 128, 128]
    output43 = builder.multiply(output37, output41)  # [1, 32, 128, 128]
    output45 = builder.add(output21, output43)  # [1, 32, 128, 128]
    output47 = builder.squeeze(output45, 0)  # [32, 128, 128]

    output49 = builder.matmul(output1, arg12)  # [128, 4096]
    output51 = builder.reshape(output49, (1, 128, 32, 128))  # [1, 128, 32, 128]
    output53 = builder.transpose(output51, -3, -2)  # [1, 32, 128, 128]
    output55 = builder.multiply(output53, output19)  # [1, 32, 128, 128]
    output57 = builder.transpose(output53, -2, -1)  # [1, 32, 128, 128]
    output59 = builder.matmul(arg7, output57)  # [1, 32, 64, 128]
    output61 = builder.transpose(output59, -2, -1)  # [1, 32, 128, 64]
    output63 = builder.multiply(output61, arg8)  # [1, 32, 128, 64]
    output65 = builder.transpose(output53, -2, -1)  # [1, 32, 128, 128]
    output67 = builder.matmul(arg9, output65)  # [1, 32, 64, 128]
    output69 = builder.transpose(output67, -2, -1)  # [1, 32, 128, 64]
    output71 = builder.concat([output63, output69], -1)  # [1, 32, 128, 128]
    output73 = builder.multiply(output71, output41)  # [1, 32, 128, 128]
    output75 = builder.add(output55, output73)  # [1, 32, 128, 128]
    output77 = builder.squeeze(output75, 0)  # [32, 128, 128]

    output79 = builder.transpose(output77, -2, -1)  # [32, 128, 128]
    output81 = builder.matmul(output47, output79)  # [32, 128, 128]
    output83 = builder.unsqueeze(output81, 0)  # [1, 32, 128, 128]
    # output85 = builder.multiply(output83, arg10)  # [1, 32, 128, 128]
    # output87 = builder.add(output85, arg1)  # [1, 32, 128, 128]
    # output89 = builder.softmax(output87, -1)  # [1, 32, 128, 128]
    # output91 = builder.squeeze(output89, 0)  # [32, 128, 128]
    # output93 = builder.matmul(output1, arg13)  # [128, 4096]
    # output95 = builder.reshape(output93, (1, 128, 32, 128))  # [1, 128, 32, 128]
    # output97 = builder.transpose(output95, -3, -2)  # [1, 32, 128, 128]
    # output99 = builder.transpose(output97, -2, -1)  # [1, 32, 128, 128]
    # output101 = builder.squeeze(output99, 0)  # [32, 128, 128]
    # output103 = builder.transpose(output101, -2, -1)  # [32, 128, 128]
    # output105 = builder.matmul(output91, output103)  # [32, 128, 128]
    # output107 = builder.unsqueeze(output105, 0)  # [1, 32, 128, 128]
    # output109 = builder.transpose(output107, -3, -2)  # [1, 128, 32, 128]
    # output111 = builder.reshape(output109, (128, 4096))  # [128, 4096]
    # output113 = builder.matmul(output111, arg14)  # [128, 4096]
    # output115 = builder.unsqueeze(output113, 0)  # [1, 128, 4096]

    save_all_output_goldens([output83], builder=builder)
    return output83


@compile_to_flatbuffer(
    [
        (1, 128, 4096),  # arg0
        (1, 1, 128, 128),  # arg1
        (1, 128),  # arg2
        (1, 64, 1),  # arg3
        (1, 32, 64, 128),  # arg4
        (1, 1),  # arg5
        (1, 32, 64, 128),  # arg6
        (1, 32, 64, 128),  # arg7
        (1, 1),  # arg8
        (1, 32, 64, 128),  # arg9
        (1, 1),  # arg10
        (4096, 4096),  # arg11
        (4096, 4096),  # arg12
        # (4096, 4096),  # arg13
        # (4096, 4096),  # arg14
    ],
    targets=["ttnn"],
    mesh_shape=[1, 2],
    module_dump=True,
)
def test_llama_attention_multidevice(
    arg0: Operand,
    arg1: Operand,
    arg2: Operand,
    arg3: Operand,
    arg4: Operand,
    arg5: Operand,
    arg6: Operand,
    arg7: Operand,
    arg8: Operand,
    arg9: Operand,
    arg10: Operand,
    arg11: Operand,
    arg12: Operand,
    # arg13: Operand,
    # arg14: Operand,
    builder: TTIRBuilder,
):
    set_graph_goldens(builder)
    output1 = builder.squeeze(arg0, 0)  # [128, 4096]
    output1 = full_to_shard_device(output1, builder, 1)  # [128, 2048]
    arg11 = full_to_shard_device(arg11, builder, 0)  # [2048, 4096]
    output3 = builder.matmul(output1, arg11)  # [128, 4096]
    output3 = builder.all_reduce(
        output3, reduce_type="#tt.reduce_type<sum>", cluster_axis=1
    )  # [128, 4096]
    output5 = builder.reshape(output3, (1, 128, 32, 128))  # [1, 128, 32, 128]
    output7 = builder.transpose(output5, -3, -2)  # [1, 32, 128, 128]
    output7 = shard_to_full_replicate(output7, builder)  # [1, 32, 128, 128]
    output7 = full_to_shard_device(output7, builder, 3)  # [1, 32, 128, 64]
    output9 = builder.unsqueeze(arg2, 1)  # [1, 1, 128]
    output9 = full_to_shard_device(output9, builder, 2)  # [1, 1, 64]
    arg3 = full_to_shard_replicate(arg3, builder)  # [1, 64, 1]
    output11 = builder.matmul(arg3, output9)  # [1, 64, 64]
    output11 = builder.all_gather(
        output11,
        all_gather_dim=2,
        cluster_axis=1,
    )  # [1, 64, 128]
    output13 = builder.transpose(output11, -2, -1)  # [1, 128, 64]
    output15 = output13  # output = output15 = builder.concat([output13, output13], -1) => concat itself. ignore it. concat and shard makes identical result
    output17 = builder.cos(output15)  # [1, 128, 64]
    output19 = builder.unsqueeze(output17, 1)  # [1, 1, 128, 64]
    output21 = builder.multiply(output7, output19)  # [1, 32, 128, 64]
    output23 = builder.transpose(output7, -2, -1)  # [1, 32, 64, 128]
    arg4 = full_to_shard_device(arg4, builder, 3)  # [1, 32, 64, 64]
    output25 = builder.matmul(arg4, output23)  # [1, 32, 64, 128]
    output25 = builder.reduce_scatter(
        output25, reduce_type="#tt.reduce_type<sum>", scatter_dim=3, cluster_axis=1
    )  # [1, 32, 64, 64]
    output27 = builder.transpose(output25, -2, -1)  # [1, 32, 64, 64]
    arg5 = full_to_shard_replicate(arg5, builder)  # [1, 1]
    output29 = builder.multiply(output27, arg5)  # [1, 32, 64, 64]
    output31 = builder.transpose(output7, -2, -1)  # [1, 32, 64, 128]
    arg6 = full_to_shard_device(arg6, builder, 3)  # [1, 32, 64, 64]
    output33 = builder.matmul(arg6, output31)  # [1, 32, 64, 128]
    output33 = builder.reduce_scatter(
        output33, reduce_type="#tt.reduce_type<sum>", scatter_dim=3, cluster_axis=1
    )  # [1, 32, 64, 64]
    output35 = builder.transpose(output33, -2, -1)  # [1, 32, 64, 64]
    output35 = builder.all_gather(
        output35, all_gather_dim=2, cluster_axis=1
    )  # [1, 32, 128, 64]
    output29 = builder.all_gather(
        output29, all_gather_dim=2, cluster_axis=1
    )  # [1, 32, 128, 64]
    output37 = builder.concat([output29, output35], -1)  # [1, 32, 128, 128]
    output37 = shard_to_full_replicate(output37, builder)  # [1, 32, 128, 128]
    output37 = full_to_shard_device(output37, builder, 3)  # [1, 32, 128, 64]
    output39 = builder.sin(output15)  # [1, 128, 64]
    output41 = builder.unsqueeze(output39, 1)  # [1, 1, 128, 64]
    output43 = builder.multiply(output37, output41)  # [1, 32, 128, 64]
    output45 = builder.add(output21, output43)  # [1, 32, 128, 64]
    output47 = builder.squeeze(output45, 0)  # [32, 128, 64]
    arg12 = full_to_shard_device(arg12, builder, 0)  # [2048, 4096]
    output49 = builder.matmul(output1, arg12)  # [128, 4096]
    output49 = builder.all_reduce(
        output49, reduce_type="#tt.reduce_type<sum>", cluster_axis=1
    )  # [128, 4096]
    output51 = builder.reshape(output49, (1, 128, 32, 128))  # [1, 128, 32, 128]
    output53 = builder.transpose(output51, -3, -2)  # [1, 32, 128, 128]
    output53 = shard_to_full_replicate(output53, builder)  # [1, 32, 128, 128]
    output53 = full_to_shard_device(output53, builder, 3)  # [1, 32, 128, 64]
    output55 = builder.multiply(output53, output19)  # [1, 32, 128, 64]
    output57 = builder.transpose(output53, -2, -1)  # [1, 32, 64, 128]
    arg7 = full_to_shard_device(arg7, builder, 3)  # [1, 32, 64, 64]
    output59 = builder.matmul(arg7, output57)  # [1, 32, 64, 128]
    output59 = builder.reduce_scatter(
        output59, reduce_type="#tt.reduce_type<sum>", scatter_dim=3, cluster_axis=1
    )  # [1, 32, 64, 64]
    output61 = builder.transpose(output59, -2, -1)  # [1, 32, 64, 64]
    arg8 = full_to_shard_replicate(arg8, builder)  # [1, 1]
    output63 = builder.multiply(output61, arg8)  # [1, 32, 64, 64]
    output65 = builder.transpose(output53, -2, -1)  # [1, 32, 64, 128]
    arg9 = full_to_shard_device(arg9, builder, 3)  # [1, 32, 64, 64]
    output67 = builder.matmul(arg9, output65)  # [1, 32, 64, 128]
    output67 = builder.reduce_scatter(
        output67, reduce_type="#tt.reduce_type<sum>", scatter_dim=3, cluster_axis=1
    )  # [1, 32, 64, 64]
    output69 = builder.transpose(output67, -2, -1)  # [1, 32, 64, 64]
    output63 = builder.all_gather(
        output63, all_gather_dim=2, cluster_axis=1
    )  # [1, 32, 128, 64]
    output69 = builder.all_gather(
        output69, all_gather_dim=2, cluster_axis=1
    )  # [1, 32, 128, 64]
    output71 = builder.concat([output63, output69], -1)  # [1, 32, 128, 128]
    output71 = shard_to_full_replicate(output71, builder)  # [1, 32, 128, 128]
    output71 = full_to_shard_device(output71, builder, 3)  # [1, 32, 128, 64]
    output73 = builder.multiply(output71, output41)  # [1, 32, 128, 64]
    output75 = builder.add(output55, output73)  # [1, 32, 128, 64]
    output77 = builder.squeeze(output75, 0)  # [32, 128, 64]
    output79 = builder.transpose(output77, -2, -1)  # [32, 64, 128]
    # testing start
    def single(output47, output79, builder):
        output47 = shard_to_full_device(output47, builder, 2)
        output79 = shard_to_full_device(output79, builder, 1)
        output81 = builder.matmul(output47, output79)  # [32, 128, 128]
        output83 = builder.unsqueeze(output81, 0)  # [1, 32, 128, 128]
        return output83  # golden matched

    def inner_product(output47, output79, builder):
        output81 = builder.matmul(output47, output79)  # [32, 128, 128]
        output83 = builder.unsqueeze(output81, 0)  # [1, 32, 128, 128]
        output83 = builder.reduce_scatter(
            output83, reduce_type="#tt.reduce_type<sum>", scatter_dim=3, cluster_axis=1
        )  # [1, 32, 128, 64]
        output83 = shard_to_full_device(output83, builder, 3)
        return output83  # mismatch actual_pcc=0.6154692030250296

    def inner_product2(output47, output79, builder):
        output81 = builder.matmul(output47, output79)  # [32, 128, 128]
        output83 = builder.all_reduce(
            output81, reduce_type="#tt.reduce_type<sum>", cluster_axis=1
        )  # [1, 32, 128, 64]
        output83 = shard_to_full_replicate(output83, builder)
        output83 = builder.unsqueeze(output83, 0)  # [1, 32, 128, 128]
        return output83  # mismatch actual_pcc=0.6154692030250296

    def inner_product_test(output47, output79, builder):
        output81 = builder.matmul(output47, output79)  # [32, 128, 128]
        output83 = shard_to_full_device(output81, builder, 2)
        output83 = builder.unsqueeze(output83, 0)  # [1, 32, 128, 128]
        return output83  # mismatch actual_pcc=0.6154692030250296

    def data_parallel(output47, output79, builder):
        output47 = shard_to_full_device(output47, builder, 2)
        output79 = shard_to_full_device(output79, builder, 1)
        output47 = full_to_shard_device(output47, builder, 0)
        output79 = full_to_shard_device(output79, builder, 0)
        output81 = builder.matmul(output47, output79)  # [32, 128, 128]
        output83 = builder.unsqueeze(output81, 0)  # [1, 32, 128, 128]
        output83 = builder.all_gather(output83, all_gather_dim=1, cluster_axis=1)
        output83 = shard_to_full_replicate(output83, builder)
        return output83  # golden matched

    def row_sharding(output47, output79, builder):
        output47 = shard_to_full_device(output47, builder, 2)
        output79 = shard_to_full_device(output79, builder, 1)
        output47 = full_to_shard_device(output47, builder, 1)  # [32, 64, 256]
        output79 = full_to_shard_replicate(output79, builder)  # [32, 256, 128]
        output81 = builder.matmul(output47, output79)  # [32, 64, 128]
        output83 = builder.unsqueeze(output81, 0)  # [1, 32, 64, 128]
        output83 = builder.all_gather(output83, all_gather_dim=2, cluster_axis=1)
        output83 = shard_to_full_replicate(output83, builder)
        return output83  # golden matched pcc=0.9999834048798768

    def col_sharding(output47, output79, builder):
        output47 = shard_to_full_device(output47, builder, 2)
        output79 = shard_to_full_device(output79, builder, 1)
        output47 = full_to_shard_replicate(output47, builder)
        output79 = full_to_shard_device(output79, builder, 2)  # [32, 256, 64]
        output81 = builder.matmul(output47, output79)  # [32, 128, 64]
        output83 = builder.unsqueeze(output81, 0)  # [1, 32, 128, 64]
        output83 = builder.all_gather(output83, all_gather_dim=3, cluster_axis=1)
        output83 = shard_to_full_replicate(output83, builder)
        return output83  # golden matched pcc=0.9999834048798768

    output83 = inner_product(output47, output79, builder)
    # output83 = builder.reduce_scatter(
    #     output83, reduce_type="#tt.reduce_type<sum>", scatter_dim=3, cluster_axis=1
    # )  # [1, 32, 128, 64]
    # testing end
    # arg10 = full_to_shard_replicate(arg10, builder)  # [1, 1]
    # output85 = builder.multiply(output83, arg10)  # [1, 32, 128, 64]
    # arg1 = full_to_shard_device(arg1, builder, 3)  # [1, 1, 128, 64]
    # output87 = builder.add(output85, arg1)  # [1, 32, 128, 64]
    # output87 = shard_to_full_device(output87, builder, 3)
    # output87 = full_to_shard_device(output87, builder, 2)
    # output89 = builder.softmax(output87, -1)  # [1, 32, 64, 128]
    # output89 = shard_to_full_device(output89, builder, 2)
    # output89 = full_to_shard_device(output89, builder, 3)
    # output91 = builder.squeeze(output89, 0)  # [32, 128, 64]
    # arg13 = full_to_shard_device(arg13, builder, 0)  # [2048, 4096]
    # output93 = builder.matmul(output1, arg13)  # [128, 4096]
    # output93 = builder.all_reduce(
    #     output93, reduce_type="#tt.reduce_type<sum>", cluster_axis=1
    # )  # [128, 4096]
    # output95 = builder.reshape(output93, (1, 128, 32, 128))  # [1, 128, 32, 128]
    # output95 = shard_to_full_replicate(output95, builder)  # [1, 128, 32, 128]
    # output95 = full_to_shard_device(output95, builder, 1)  # [1, 64, 32, 128]
    # output97 = builder.transpose(output95, -3, -2)  # [1, 32, 64, 128]
    # output99 = builder.transpose(output97, -2, -1)  # [1, 32, 128, 64]
    # output101 = builder.squeeze(output99, 0)  # [32, 128, 64]
    # output103 = builder.transpose(output101, -2, -1)  # [32, 64, 128]
    # output105 = builder.matmul(output91, output103)  # [32, 128, 128]
    # output105 = builder.all_reduce(
    #     output105, reduce_type="#tt.reduce_type<sum>", cluster_axis=1
    # )  # [32, 128, 128]
    # output107 = builder.unsqueeze(output105, 0)  # [1, 32, 128, 128]
    # output109 = builder.transpose(output107, -3, -2)  # [1, 128, 32, 128]
    # output111 = builder.reshape(output109, (128, 4096))  # [128, 4096]
    # arg14 = full_to_shard_device(arg14, builder, 1)  # [4096, 2048]
    # output113 = builder.matmul(output111, arg14)  # [128, 2048]
    # output115 = builder.unsqueeze(output113, 0)  # [1, 128, 2048]
    # output115 = shard_to_full_device(output115, builder, dim=2)  # [1, 128, 4096]
    return output83


@compile_to_flatbuffer(
    [
        (1, 128, 4096),  # arg0
        (1, 32, 64, 128),  # arg4
        (1, 1),  # arg5
        (1, 32, 64, 128),  # arg6
        (4096, 4096),  # arg11
        (1, 32, 128, 128),  # output21
        (1, 1, 128, 128),  # output41
    ],
    targets=["ttnn"],
    module_dump=True,
)
def test_llama_attention_query(
    arg0: Operand,
    arg4: Operand,
    arg5: Operand,
    arg6: Operand,
    arg11: Operand,
    output21: Operand,
    output41: Operand,
    builder: TTIRBuilder,
):
    save_all_input_goldens(
        [v for k, v in locals().items() if k != "builder"], builder=builder
    )
    output1 = builder.squeeze(arg0, 0)  # [128, 4096]

    output3 = builder.matmul(output1, arg11)  # [128, 4096]
    output5 = builder.reshape(output3, (1, 128, 32, 128))  # [1, 128, 32, 128]
    output7 = builder.transpose(output5, -3, -2)  # [1, 32, 128, 128]
    output23 = builder.transpose(output7, -2, -1)  # [1, 32, 128, 128]
    output25 = builder.matmul(arg4, output23)  # [1, 32, 64, 128]
    output27 = builder.transpose(output25, -2, -1)  # [1, 32, 128, 64]
    output29 = builder.multiply(output27, arg5)  # [1, 32, 128, 64]
    output31 = builder.transpose(output7, -2, -1)  # [1, 32, 128, 128]
    output33 = builder.matmul(arg6, output31)  # [1, 32, 64, 128]
    output35 = builder.transpose(output33, -2, -1)  # [1, 32, 128, 64]
    output37 = builder.concat([output29, output35], -1)  # [1, 32, 128, 128]
    output43 = builder.multiply(output37, output41)  # [1, 32, 128, 128]
    output45 = builder.add(output21, output43)  # [1, 32, 128, 128]
    output47 = builder.squeeze(output45, 0)  # [32, 128, 128]
    save_all_output_goldens([output47], builder=builder)
    return output47


@compile_to_flatbuffer(
    [
        (1, 128, 4096),  # arg0
        (1, 32, 64, 128),  # arg4
        (1, 1),  # arg5
        (1, 32, 64, 128),  # arg6
        (4096, 4096),  # arg11
        (1, 32, 128, 128),  # output21
        (1, 1, 128, 128),  # output41
    ],
    targets=["ttnn"],
    module_dump=True,
    mesh_shape=(1, 2),
)
def test_llama_attention_query_multidevice(
    arg0: Operand,
    arg4: Operand,
    arg5: Operand,
    arg6: Operand,
    arg11: Operand,
    output21: Operand,
    output41: Operand,
    builder: TTIRBuilder,
):
    set_graph_goldens(builder)
    output1 = builder.squeeze(arg0, 0)  # [128, 4096]

    output1 = builder.squeeze(arg0, 0)  # [128, 4096]
    output1 = full_to_shard_device(output1, builder, 1)  # [128, 2048]
    arg11 = full_to_shard_device(arg11, builder, 0)  # [2048, 4096]
    output3 = builder.matmul(output1, arg11)  # [128, 4096]
    output3 = builder.all_reduce(
        output3, reduce_type="#tt.reduce_type<sum>", cluster_axis=1
    )  # [128, 4096]
    output5 = builder.reshape(output3, (1, 128, 32, 128))  # [1, 128, 32, 128]
    output7 = builder.transpose(output5, -3, -2)  # [1, 32, 128, 128]
    output7 = shard_to_full_replicate(output7, builder)  # [1, 32, 128, 128]
    output7 = full_to_shard_device(output7, builder, 3)  # [1, 32, 128, 64]

    output23 = builder.transpose(output7, -2, -1)  # [1, 32, 64, 128]
    arg4 = full_to_shard_device(arg4, builder, 3)  # [1, 32, 64, 64]
    output25 = builder.matmul(arg4, output23)  # [1, 32, 64, 128]
    output25 = builder.reduce_scatter(
        output25, reduce_type="#tt.reduce_type<sum>", scatter_dim=3, cluster_axis=1
    )  # [1, 32, 64, 64]
    output27 = builder.transpose(output25, -2, -1)  # [1, 32, 64, 64]
    arg5 = full_to_shard_replicate(arg5, builder)  # [1, 1]
    output29 = builder.multiply(output27, arg5)  # [1, 32, 64, 64]
    output31 = builder.transpose(output7, -2, -1)  # [1, 32, 64, 128]
    arg6 = full_to_shard_device(arg6, builder, 3)  # [1, 32, 64, 64]
    output33 = builder.matmul(arg6, output31)  # [1, 32, 64, 128]
    output33 = builder.reduce_scatter(
        output33, reduce_type="#tt.reduce_type<sum>", scatter_dim=3, cluster_axis=1
    )  # [1, 32, 64, 64]
    output35 = builder.transpose(output33, -2, -1)  # [1, 32, 64, 64]
    output35 = builder.all_gather(
        output35, all_gather_dim=2, cluster_axis=1
    )  # [1, 32, 128, 64]
    output29 = builder.all_gather(
        output29, all_gather_dim=2, cluster_axis=1
    )  # [1, 32, 128, 64]
    output37 = builder.concat([output29, output35], -1)  # [1, 32, 128, 128]
    output37 = shard_to_full_replicate(output37, builder)  # [1, 32, 128, 128]

    output37 = full_to_shard_device(output37, builder, 3)  # [1, 32, 128, 64]
    output41 = full_to_shard_device(output41, builder, 3)  # [1, 32, 128, 64]
    output43 = builder.multiply(output37, output41)  # [1, 32, 128, 64]
    output21 = full_to_shard_device(output21, builder, 3)  # [1, 32, 128, 64]
    output45 = builder.add(output21, output43)  # [1, 32, 128, 64]
    output47 = builder.squeeze(output45, 0)  # [32, 128, 64]
    output47 = shard_to_full_device(output47, builder, 2)
    return output47


@compile_to_flatbuffer(
    [
        (1, 128, 4096),  # arg0
        (1, 32, 64, 128),  # arg7
        (1, 1),  # arg8
        (1, 32, 64, 128),  # arg9
        (4096, 4096),  # arg12
        (1, 1, 128, 128),  # output19
        (1, 1, 128, 128),  # output41
    ],
    targets=["ttnn"],
    module_dump=True,
)
def test_llama_attention_key(
    arg0: Operand,
    arg7: Operand,
    arg8: Operand,
    arg9: Operand,
    arg12: Operand,
    output19: Operand,
    output41: Operand,
    builder: TTIRBuilder,
):
    save_all_input_goldens(
        [v for k, v in locals().items() if k != "builder"], builder=builder
    )
    output1 = builder.squeeze(arg0, 0)  # [128, 4096]

    output49 = builder.matmul(output1, arg12)  # [128, 4096]
    output51 = builder.reshape(output49, (1, 128, 32, 128))  # [1, 128, 32, 128]
    output53 = builder.transpose(output51, -3, -2)  # [1, 32, 128, 128]
    output55 = builder.multiply(output53, output19)  # [1, 32, 128, 128]
    output57 = builder.transpose(output53, -2, -1)  # [1, 32, 128, 128]
    output59 = builder.matmul(arg7, output57)  # [1, 32, 64, 128]
    output61 = builder.transpose(output59, -2, -1)  # [1, 32, 128, 64]
    output63 = builder.multiply(output61, arg8)  # [1, 32, 128, 64]
    output65 = builder.transpose(output53, -2, -1)  # [1, 32, 128, 128]
    output67 = builder.matmul(arg9, output65)  # [1, 32, 64, 128]
    output69 = builder.transpose(output67, -2, -1)  # [1, 32, 128, 64]
    output71 = builder.concat([output63, output69], -1)  # [1, 32, 128, 128]
    output73 = builder.multiply(output71, output41)  # [1, 32, 128, 128]
    output75 = builder.add(output55, output73)  # [1, 32, 128, 128]
    output77 = builder.squeeze(output75, 0)  # [32, 128, 128]
    save_all_output_goldens([output77], builder=builder)
    return output77


@compile_to_flatbuffer(
    [
        (1, 128, 4096),  # arg0
        (1, 32, 64, 128),  # arg7
        (1, 1),  # arg8
        (1, 32, 64, 128),  # arg9
        (4096, 4096),  # arg12
        (1, 1, 128, 128),  # output19
        (1, 1, 128, 128),  # output41
    ],
    targets=["ttnn"],
    module_dump=True,
    mesh_shape=(1, 2),
)
def test_llama_attention_key_multidevice(
    arg0: Operand,
    arg7: Operand,
    arg8: Operand,
    arg9: Operand,
    arg12: Operand,
    output19: Operand,
    output41: Operand,
    builder: TTIRBuilder,
):
    set_graph_goldens(builder)
    output1 = builder.squeeze(arg0, 0)  # [128, 4096]
    output1 = full_to_shard_device(output1, builder, 1)  # [128, 2048]
    arg12 = full_to_shard_device(arg12, builder, 0)  # [2048, 4096]
    output49 = builder.matmul(output1, arg12)  # [128, 4096]
    output49 = builder.all_reduce(
        output49, reduce_type="#tt.reduce_type<sum>", cluster_axis=1
    )  # [128, 4096]
    output51 = builder.reshape(output49, (1, 128, 32, 128))  # [1, 128, 32, 128]
    output53 = builder.transpose(output51, -3, -2)  # [1, 32, 128, 128]
    output53 = shard_to_full_replicate(output53, builder)  # [1, 32, 128, 128]
    output53 = full_to_shard_device(output53, builder, 3)  # [1, 32, 128, 64]
    output19 = full_to_shard_device(output19, builder, 3)  # [1, 32, 64, 64]
    output55 = builder.multiply(output53, output19)  # [1, 32, 128, 64]
    output57 = builder.transpose(output53, -2, -1)  # [1, 32, 64, 128]
    arg7 = full_to_shard_device(arg7, builder, 3)  # [1, 32, 64, 64]
    output59 = builder.matmul(arg7, output57)  # [1, 32, 64, 128]
    output59 = builder.reduce_scatter(
        output59, reduce_type="#tt.reduce_type<sum>", scatter_dim=3, cluster_axis=1
    )  # [1, 32, 64, 64]
    output61 = builder.transpose(output59, -2, -1)  # [1, 32, 64, 64]
    arg8 = full_to_shard_replicate(arg8, builder)  # [1, 1]
    output63 = builder.multiply(output61, arg8)  # [1, 32, 64, 64]
    output65 = builder.transpose(output53, -2, -1)  # [1, 32, 64, 128]
    arg9 = full_to_shard_device(arg9, builder, 3)  # [1, 32, 64, 64]
    output67 = builder.matmul(arg9, output65)  # [1, 32, 64, 128]
    output67 = builder.reduce_scatter(
        output67, reduce_type="#tt.reduce_type<sum>", scatter_dim=3, cluster_axis=1
    )  # [1, 32, 64, 64]
    output69 = builder.transpose(output67, -2, -1)  # [1, 32, 64, 64]
    output63 = builder.all_gather(
        output63, all_gather_dim=2, cluster_axis=1
    )  # [1, 32, 128, 64]
    output69 = builder.all_gather(
        output69, all_gather_dim=2, cluster_axis=1
    )  # [1, 32, 128, 64]
    output71 = builder.concat([output63, output69], -1)  # [1, 32, 128, 128]
    output71 = shard_to_full_replicate(output71, builder)  # [1, 32, 128, 128]
    output71 = full_to_shard_device(output71, builder, 3)  # [1, 32, 128, 64]
    output41 = full_to_shard_device(output41, builder, 3)  # [1, 32, 64, 64]
    output73 = builder.multiply(output71, output41)  # [1, 32, 128, 64]
    output75 = builder.add(output55, output73)  # [1, 32, 128, 64]
    output77 = builder.squeeze(output75, 0)  # [32, 128, 64]
    output77 = shard_to_full_device(output77, builder, 2)
    return output77


@compile_to_flatbuffer(
    [
        (1, 128),  # arg2
        (1, 64, 1),  # arg3
        (1, 32, 128, 128),  # output7
    ],
    targets=["ttnn"],
    module_dump=True,
)
def test_llama_attention_RoPE(
    arg2: Operand,
    arg3: Operand,
    output7: Operand,
    builder: TTIRBuilder,
):
    save_all_input_goldens(
        [v for k, v in locals().items() if k != "builder"], builder=builder
    )
    output9 = builder.unsqueeze(arg2, 1)  # [1, 1, 128]
    output11 = builder.matmul(arg3, output9)  # [1, 64, 128]
    output13 = builder.transpose(output11, -2, -1)  # [1, 128, 64]
    output15 = builder.concat([output13, output13], -1)  # [1, 128, 128]
    output17 = builder.cos(output15)  # [1, 128, 128]
    output19 = builder.unsqueeze(output17, 1)  # [1, 1, 128, 128]
    output21 = builder.multiply(output7, output19)  # [1, 32, 128, 128]
    output39 = builder.sin(output15)  # [1, 128, 128]
    output41 = builder.unsqueeze(output39, 1)  # [1, 1, 128, 128]
    save_all_output_goldens([output41], builder=builder)
    return output41


@compile_to_flatbuffer(
    [
        (1, 128),  # arg2
        (1, 64, 1),  # arg3
        (1, 32, 128, 128),  # output7
    ],
    targets=["ttnn"],
    module_dump=True,
    mesh_shape=(1, 2),
)
def test_llama_attention_RoPE_multidevice(
    arg2: Operand,
    arg3: Operand,
    output7: Operand,
    builder: TTIRBuilder,
):
    set_graph_goldens(builder)
    output9 = builder.unsqueeze(arg2, 1)  # [1, 1, 128]
    output9 = full_to_shard_device(output9, builder, 2)  # [1, 1, 64]
    arg3 = full_to_shard_replicate(arg3, builder)  # [1, 64, 1]
    output11 = builder.matmul(arg3, output9)  # [1, 64, 64]
    output11 = builder.all_gather(
        output11,
        all_gather_dim=2,
        cluster_axis=1,
    )  # [1, 64, 128]
    output13 = builder.transpose(output11, -2, -1)  # [1, 128, 64]
    output15 = output13  # output = output15 = builder.concat([output13, output13], -1) => concat itself. ignore it. concat and shard makes identical result
    output17 = builder.cos(output15)  # [1, 128, 64]
    output19 = builder.unsqueeze(output17, 1)  # [1, 1, 128, 64]
    output7 = full_to_shard_device(output7, builder, 3)  # [1, 32, 128, 64]
    output21 = builder.multiply(output7, output19)  # [1, 32, 128, 64]
    output21 = shard_to_full_device(output21, builder, 3)
    output39 = builder.sin(output15)  # [1, 128, 64]
    output41 = builder.unsqueeze(output39, 1)  # [1, 1, 128, 64]
    output41 = shard_to_full_device(output41, builder, 3)
    return output41


@compile_to_flatbuffer(
    [
        (1, 128, 4096),  # arg0
        (1, 1, 128, 128),  # arg1
        (1, 1),  # arg10
        (4096, 4096),  # arg13
        (4096, 4096),  # arg14
        (32, 128, 128),  # output47
        (32, 128, 128),  # output77
    ],
    targets=["ttnn"],
    module_dump=True,
)
def test_llama_attention_value_scaled_dot(
    arg0: Operand,
    arg1: Operand,
    arg10: Operand,
    arg13: Operand,
    arg14: Operand,
    output47: Operand,
    output77: Operand,
    builder: TTIRBuilder,
):
    save_all_input_goldens(
        [v for k, v in locals().items() if k != "builder"], builder=builder
    )
    output1 = builder.squeeze(arg0, 0)  # [128, 4096]
    output79 = builder.transpose(output77, -2, -1)  # [32, 128, 128]
    output81 = builder.matmul(output47, output79)  # [32, 128, 128]
    output83 = builder.unsqueeze(output81, 0)  # [1, 32, 128, 128]
    output85 = builder.multiply(output83, arg10)  # [1, 32, 128, 128]
    output87 = builder.add(output85, arg1)  # [1, 32, 128, 128]
    output89 = builder.softmax(output87, -1)  # [1, 32, 128, 128]
    output91 = builder.squeeze(output89, 0)  # [32, 128, 128]
    output93 = builder.matmul(output1, arg13)  # [128, 4096]
    output95 = builder.reshape(output93, (1, 128, 32, 128))  # [1, 128, 32, 128]
    output97 = builder.transpose(output95, -3, -2)  # [1, 32, 128, 128]
    output99 = builder.transpose(output97, -2, -1)  # [1, 32, 128, 128]
    output101 = builder.squeeze(output99, 0)  # [32, 128, 128]
    output103 = builder.transpose(output101, -2, -1)  # [32, 128, 128]
    output105 = builder.matmul(output91, output103)  # [32, 128, 128]
    output107 = builder.unsqueeze(output105, 0)  # [1, 32, 128, 128]
    output109 = builder.transpose(output107, -3, -2)  # [1, 128, 32, 128]
    output111 = builder.reshape(output109, (128, 4096))  # [128, 4096]
    output113 = builder.matmul(output111, arg14)  # [128, 4096]
    output115 = builder.unsqueeze(output113, 0)  # [1, 128, 4096]
    save_all_output_goldens([output115], builder=builder)
    return output115


@compile_to_flatbuffer(
    [
        (1, 128, 4096),  # arg0
        (1, 1, 128, 128),  # arg1
        (1, 1),  # arg10
        (4096, 4096),  # arg13
        (4096, 4096),  # arg14
        (32, 128, 128),  # output47
        (32, 128, 128),  # output77
    ],
    targets=["ttnn"],
    module_dump=True,
    mesh_shape=(1, 2),
)
def test_llama_attention_value_scaled_dot_multidevice(
    arg0: Operand,
    arg1: Operand,
    arg10: Operand,
    arg13: Operand,
    arg14: Operand,
    output47: Operand,
    output77: Operand,
    builder: TTIRBuilder,
):
    set_graph_goldens(builder)
    output1 = builder.squeeze(arg0, 0)  # [128, 4096]
    output77 = full_to_shard_device(output77, builder, 2)
    output79 = builder.transpose(output77, -2, -1)  # [32, 64, 128]
    output47 = full_to_shard_device(output47, builder, 2)
    output81 = builder.matmul(output47, output79)  # [32, 128, 128]
    output83 = builder.unsqueeze(output81, 0)  # [1, 32, 128, 128]
    output83 = builder.reduce_scatter(
        output83, reduce_type="#tt.reduce_type<sum>", scatter_dim=3, cluster_axis=1
    )  # [1, 32, 128, 64]
    arg10 = full_to_shard_replicate(arg10, builder)  # [1, 1]
    output85 = builder.multiply(output83, arg10)  # [1, 32, 128, 64]
    arg1 = full_to_shard_device(arg1, builder, 3)  # [1, 1, 128, 64]
    output87 = builder.add(output85, arg1)  # [1, 32, 128, 64]
    output87 = shard_to_full_device(output87, builder, 3)  # [1, 32, 128, 128]
    output87 = full_to_shard_device(output87, builder, 2)  # [1, 32, 64, 128]
    output89 = builder.softmax(output87, -1)  # [1, 32, 64, 128]
    output89 = shard_to_full_device(output89, builder, 2)  # [32, 128, 128]
    output89 = full_to_shard_device(output89, builder, 3)  # [32, 128, 64]
    output91 = builder.squeeze(output89, 0)  # [32, 64, 128]
    arg13 = full_to_shard_device(arg13, builder, 0)  # [2048, 4096]
    output1 = full_to_shard_device(output1, builder, 1)  # [128, 2048]
    output93 = builder.matmul(output1, arg13)  # [128, 4096]
    output93 = builder.all_reduce(
        output93, reduce_type="#tt.reduce_type<sum>", cluster_axis=1
    )  # [128, 4096]
    output95 = builder.reshape(output93, (1, 128, 32, 128))  # [1, 128, 32, 128]
    output95 = shard_to_full_replicate(output95, builder)  # [1, 128, 32, 128]
    output95 = full_to_shard_device(output95, builder, 1)  # [1, 64, 32, 128]
    output97 = builder.transpose(output95, -3, -2)  # [1, 32, 64, 128]
    output99 = builder.transpose(output97, -2, -1)  # [1, 32, 128, 64]
    output101 = builder.squeeze(output99, 0)  # [32, 128, 64]
    output103 = builder.transpose(output101, -2, -1)  # [32, 64, 128]
    output105 = builder.matmul(output91, output103)  # [32, 128, 128]
    output105 = builder.all_reduce(
        output105, reduce_type="#tt.reduce_type<sum>", cluster_axis=1
    )  # [32, 128, 128]
    output107 = builder.unsqueeze(output105, 0)  # [1, 32, 128, 128]
    output109 = builder.transpose(output107, -3, -2)  # [1, 128, 32, 128]
    output111 = builder.reshape(output109, (128, 4096))  # [128, 4096]
    arg14 = full_to_shard_device(arg14, builder, 1)  # [4096, 2048]
    output113 = builder.matmul(output111, arg14)  # [128, 2048]
    output115 = builder.unsqueeze(output113, 0)  # [1, 128, 2048]
    output115 = shard_to_full_device(output115, builder, dim=2)  # [1, 128, 4096]
    return output115


if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Run TTIR Builder Model tests")
    parser.add_argument(
        "--path",
        type=str,
        help="Optional output path for the flatbuffer. Creates path if supplied path doesn't exist",
    )
    args = parser.parse_args()

    if args.path and os.path.exists(args.path):
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        set_output_path(args.path)

    test_functions = inspect.getmembers(
        inspect.getmodule(inspect.currentframe()), inspect.isfunction
    )

    test_llama_attention_query()
    test_llama_attention_query_multidevice()
    test_llama_attention_key()
    test_llama_attention_key_multidevice()
    test_llama_attention_RoPE()
    test_llama_attention_RoPE_multidevice()
    test_llama_attention_value_scaled_dot()
    test_llama_attention_value_scaled_dot_multidevice()
    test_llama_attention()
    test_llama_attention_multidevice()
