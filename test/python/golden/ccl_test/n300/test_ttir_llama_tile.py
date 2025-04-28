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
        (1, 256, 16384),  # arg0
        # (1, 1, 256, 256),  # arg1
        (1, 256),  # arg2
        (1, 256, 1),  # arg3
        # (1, 32, 256, 128),  # arg4
        # (1, 1),  # arg5
        # (1, 32, 256, 128),  # arg6
        # (1, 32, 256, 128),  # arg7
        # (1, 1),  # arg8
        # (1, 32, 256, 128),  # arg9
        # (1, 1),  # arg10
        (16384, 16384),  # arg11
        # (16384, 16384),  # arg12
        # (16384, 16384),  # arg13
        # (16384, 16384),  # arg14
    ],
    targets=["ttnn"],
    module_dump=True,
)
def test_llama_attention(
    arg0: Operand,
    # arg1: Operand,
    arg2: Operand,
    arg3: Operand,
    # arg4: Operand,
    # arg5: Operand,
    # arg6: Operand,
    # arg7: Operand,
    # arg8: Operand,
    # arg9: Operand,
    # arg10: Operand,
    arg11: Operand,
    # arg12: Operand,
    # arg13: Operand,
    # arg14: Operand,
    builder: TTIRBuilder,
):
    save_all_input_goldens(
        [v for k, v in locals().items() if k != "builder"], builder=builder
    )

    output = output1 = builder.squeeze(arg0, 0)  # [256, 16384]
    output = output3 = builder.matmul(output1, arg11)  # [256, 16384]
    output = output5 = builder.reshape(output3, (1, 256, 32, 512))  # [1, 256, 32, 128]
    output = output7 = builder.transpose(output5, -3, -2)  # [1, 32, 256, 128]
    output = output9 = builder.unsqueeze(arg2, 1)  # [1, 1, 256]
    output = output11 = builder.matmul(arg3, output9)  # [1, 256, 256]
    output = output13 = builder.transpose(output11, -2, -1)  # [1, 256, 256]
    output = output15 = builder.concat([output13, output13], -1)  # [1, 256, 512]
    output = output17 = builder.cos(output15)  # [1, 256, 512]
    output = output19 = builder.unsqueeze(output17, 1)  # [1, 1, 256, 512]
    output = output21 = builder.multiply(output7, output19)  # [1, 32, 256, 512]
    # output = output23 = builder.transpose(output7, -2, -1)    # [1, 32, 128, 256]
    # output = output25 = builder.matmul(arg4, output23)    # [1, 32, 256, 256]
    # output = output27 = builder.transpose(output25, -2, -1)    # [1, 32, 256, 256]
    # output = output29 = builder.multiply(output27, arg5)    # [1, 32, 256, 256]
    # output = output31 = builder.transpose(output7, -2, -1)    # [1, 32, 128, 256]
    # output = output33 = builder.matmul(arg6, output31)    # [1, 32, 256, 256]
    # output = output35 = builder.transpose(output33, -2, -1)    # [1, 32, 256, 256]
    # output = output37 = builder.concat([output29, output35], -1)    # [1, 32, 256, 128]
    # output = output39 = builder.sin(output15)    # [1, 256, 128]
    # output = output41 = builder.unsqueeze(output39, 1)    # [1, 1, 256, 128]
    # output = output43 = builder.multiply(output37, output41)    # [1, 32, 256, 128]
    # output = output45 = builder.add(output21, output43)    # [1, 32, 256, 128]
    # output = output47 = builder.squeeze(output45, 0)    # [32, 256, 128]
    # output = output49 = builder.matmul(output1, arg12)    # [256, 16384]
    # output = output51 = builder.reshape(output49, (1, 256, 32, 128))    # [1, 256, 32, 128]
    # output = output53 = builder.transpose(output51, -3, -2)    # [1, 32, 256, 128]
    # output = output55 = builder.multiply(output53, output19)    # [1, 32, 256, 128]
    # output = output57 = builder.transpose(output53, -2, -1)    # [1, 32, 128, 256]
    # output = output59 = builder.matmul(arg7, output57)    # [1, 32, 256, 256]
    # output = output61 = builder.transpose(output59, -2, -1)    # [1, 32, 256, 256]
    # output = output63 = builder.multiply(output61, arg8)    # [1, 32, 256, 256]
    # output = output65 = builder.transpose(output53, -2, -1)    # [1, 32, 128, 256]
    # output = output67 = builder.matmul(arg9, output65)    # [1, 32, 256, 256]
    # output = output69 = builder.transpose(output67, -2, -1)    # [1, 32, 256, 256]
    # output = output71 = builder.concat([output63, output69], -1)    # [1, 32, 256, 128]
    # output = output73 = builder.multiply(output71, output41)    # [1, 32, 256, 128]
    # output = output75 = builder.add(output55, output73)    # [1, 32, 256, 128]
    # output = output77 = builder.squeeze(output75, 0)    # [32, 256, 128]
    # output = output79 = builder.transpose(output77, -2, -1)    # [32, 128, 256]
    # output = output81 = builder.matmul(output47, output79)    # [32, 256, 256]
    # output = output83 = builder.unsqueeze(output81, 0)    # [1, 32, 256, 256]
    # pcc drop point
    # output = output85 = builder.multiply(output83, arg10)    # [1, 32, 256, 256]
    # output = output87 = builder.add(output85, arg1)    # [1, 32, 256, 256]
    # output = output89 = builder.softmax(output87, -1)    # [1, 32, 256, 256]
    # output = output91 = builder.squeeze(output89, 0)    # [32, 256, 256]
    # output = output93 = builder.matmul(output1, arg13)    # [256, 16384]
    # output = output95 = builder.reshape(output93, (1, 256, 32, 128))    # [1, 256, 32, 128]
    # output = output97 = builder.transpose(output95, -3, -2)    # [1, 32, 256, 128]
    # output = output99 = builder.transpose(output97, -2, -1)    # [1, 32, 128, 256]
    # output = output101 = builder.squeeze(output99, 0)    # [32, 128, 256]
    # output = output103 = builder.transpose(output101, -2, -1)    # [32, 256, 128]
    # output = output105 = builder.matmul(output91, output103)    # [32, 256, 128]
    # output = output107 = builder.unsqueeze(output105, 0)    # [1, 32, 256, 128]
    # output = output109 = builder.transpose(output107, -3, -2)    # [1, 256, 32, 128]
    # output = output111 = builder.reshape(output109, (256, 16384))    # [256, 16384]
    # output = output113 = builder.matmul(output111, arg14)    # [256, 16384]
    # output = output115 = builder.unsqueeze(output113, 0)    # [1, 256, 16384]

    save_all_output_goldens([output], builder=builder)
    return output


@compile_to_flatbuffer(
    [
        (1, 256, 16384),  # arg0
        # (1, 1, 256, 256),  # arg1
        (1, 256),  # arg2
        (1, 256, 1),  # arg3
        # (1, 32, 256, 128),  # arg4
        # (1, 1),  # arg5
        # (1, 32, 256, 128),  # arg6
        # (1, 32, 256, 128),  # arg7
        # (1, 1),  # arg8
        # (1, 32, 256, 128),  # arg9
        # (1, 1),  # arg10
        (16384, 16384),  # arg11
        # (16384, 16384),  # arg12
        # (16384, 16384),  # arg13
        # (16384, 16384),  # arg14
    ],
    targets=["ttnn"],
    mesh_shape=[1, 2],
    module_dump=True,
)
def test_llama_attention_multidevice(
    arg0: Operand,
    # arg1: Operand,
    arg2: Operand,
    arg3: Operand,
    # arg4: Operand,
    # arg5: Operand,
    # arg6: Operand,
    # arg7: Operand,
    # arg8: Operand,
    # arg9: Operand,
    # arg10: Operand,
    arg11: Operand,
    # arg12: Operand,
    # arg13: Operand,
    # arg14: Operand,
    builder: TTIRBuilder,
):
    set_graph_goldens(builder)
    output = output1 = builder.squeeze(arg0, 0)  # [256, 16384]
    output1 = full_to_shard_device(output1, builder, 1)
    arg11 = full_to_shard_device(arg11, builder, 0)
    output = output3 = builder.matmul(output1, arg11)  # [256, 16384]
    output = output3 = builder.all_reduce(
        output3,
        reduce_type="#tt.reduce_type<sum>",
        cluster_axis=1,
    )
    # output = output3 = builder.reduce_scatter(
    #     output3,
    #     reduce_type="#tt.reduce_type<sum>",
    #     scatter_dim=1,
    #     cluster_axis=1,
    # )
    output = output5 = builder.reshape(output3, (1, 256, 32, 512))  # [1, 256, 32, 512]
    output = output7 = builder.transpose(output5, -3, -2)  # [1, 32, 256, 512]
    output = output7 = shard_to_full_replicate(output7, builder)
    output = output7 = full_to_shard_device(output7, builder, 3)
    output = output9 = builder.unsqueeze(arg2, 1)  # [1, 1, 256]
    output9 = full_to_shard_device(output9, builder, 2)
    arg3 = full_to_shard_replicate(arg3, builder)
    output = output11 = builder.matmul(arg3, output9)  # [1, 256, 256]
    output = output11 = builder.all_gather(
        output11,
        all_gather_dim=2,
        cluster_axis=1,
    )
    output = output13 = builder.transpose(output11, -2, -1)  # [1, 256, 256]
    # output = output15 = builder.concat([output13, output13], -1)    # [1, 256, 512] => concat itself. ignore it. concat and shard makes identical result
    output = output15 = output13
    # output = output15 = builder.all_gather(output13, all_gather_dim = 2, cluster_axis = 1)
    # output = shard_to_full_replicate(output, builder)
    output = output17 = builder.cos(output13)  # [1, 256, 128]
    output = output19 = builder.unsqueeze(output17, 1)  # [1, 1, 256, 128]
    output = output21 = builder.multiply(output7, output19)  # [1, 32, 256, 128]
    output = shard_to_full_device(output, builder, dim=3)
    # output = output23 = builder.transpose(output7, -2, -1)    # [1, 32, 128, 256]
    # output = output25 = builder.matmul(arg4, output23)    # [1, 32, 256, 256]
    # output = output27 = builder.transpose(output25, -2, -1)    # [1, 32, 256, 256]
    # output = output29 = builder.multiply(output27, arg5)    # [1, 32, 256, 256]
    # output = output31 = builder.transpose(output7, -2, -1)    # [1, 32, 128, 256]
    # output = output33 = builder.matmul(arg6, output31)    # [1, 32, 256, 256]
    # output = output35 = builder.transpose(output33, -2, -1)    # [1, 32, 256, 256]
    # output = output37 = builder.concat([output29, output35], -1)    # [1, 32, 256, 128]
    # output = output39 = builder.sin(output15)    # [1, 256, 128]
    # output = output41 = builder.unsqueeze(output39, 1)    # [1, 1, 256, 128]
    # output = output43 = builder.multiply(output37, output41)    # [1, 32, 256, 128]
    # output = output45 = builder.add(output21, output43)    # [1, 32, 256, 128]
    # output = output47 = builder.squeeze(output45, 0)    # [32, 256, 128]
    # output = output49 = builder.matmul(output1, arg12)    # [256, 16384]
    # output = output51 = builder.reshape(output49, (1, 256, 32, 128))    # [1, 256, 32, 128]
    # output = output53 = builder.transpose(output51, -3, -2)    # [1, 32, 256, 128]
    # output = output55 = builder.multiply(output53, output19)    # [1, 32, 256, 128]
    # output = output57 = builder.transpose(output53, -2, -1)    # [1, 32, 128, 256]
    # output = output59 = builder.matmul(arg7, output57)    # [1, 32, 256, 256]
    # output = output61 = builder.transpose(output59, -2, -1)    # [1, 32, 256, 256]
    # output = output63 = builder.multiply(output61, arg8)    # [1, 32, 256, 256]
    # output = output65 = builder.transpose(output53, -2, -1)    # [1, 32, 128, 256]
    # output = output67 = builder.matmul(arg9, output65)    # [1, 32, 256, 256]
    # output = output69 = builder.transpose(output67, -2, -1)    # [1, 32, 256, 256]
    # output = output71 = builder.concat([output63, output69], -1)    # [1, 32, 256, 128]
    # output = output73 = builder.multiply(output71, output41)    # [1, 32, 256, 128]
    # output = output75 = builder.add(output55, output73)    # [1, 32, 256, 128]
    # output = output77 = builder.squeeze(output75, 0)    # [32, 256, 128]
    # output = output79 = builder.transpose(output77, -2, -1)    # [32, 128, 256]
    # output = output81 = builder.matmul(output47, output79)    # [32, 256, 256]
    # output = output83 = builder.unsqueeze(output81, 0)    # [1, 32, 256, 256]
    # pcc drop point
    # output = output85 = builder.multiply(output83, arg10)    # [1, 32, 256, 256]
    # output = output87 = builder.add(output85, arg1)    # [1, 32, 256, 256]
    # output = output89 = builder.softmax(output87, -1)    # [1, 32, 256, 256]
    # output = output91 = builder.squeeze(output89, 0)    # [32, 256, 256]
    # output = output93 = builder.matmul(output1, arg13)    # [256, 16384]
    # output = output95 = builder.reshape(output93, (1, 256, 32, 128))    # [1, 256, 32, 128]
    # output = output97 = builder.transpose(output95, -3, -2)    # [1, 32, 256, 128]
    # output = output99 = builder.transpose(output97, -2, -1)    # [1, 32, 128, 256]
    # output = output101 = builder.squeeze(output99, 0)    # [32, 128, 256]
    # output = output103 = builder.transpose(output101, -2, -1)    # [32, 256, 128]
    # output = output105 = builder.matmul(output91, output103)    # [32, 256, 128]
    # output = output107 = builder.unsqueeze(output105, 0)    # [1, 32, 256, 128]
    # output = output109 = builder.transpose(output107, -3, -2)    # [1, 256, 32, 128]
    # output = output111 = builder.reshape(output109, (256, 16384))    # [256, 16384]
    # output = output113 = builder.matmul(output111, arg14)    # [256, 16384]
    # output = output115 = builder.unsqueeze(output113, 0)    # [1, 256, 16384]

    return output


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

    for function_name, func in test_functions:
        if function_name.startswith("test_"):
            func()
