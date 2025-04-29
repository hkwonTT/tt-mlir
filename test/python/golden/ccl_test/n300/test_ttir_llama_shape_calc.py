# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: SYSTEM_DESC_PATH=%system_desc_path% %python %s

import inspect

from ttmlir.test_utils import compile_to_flatbuffer, set_output_path
from ttmlir.ttir_builder import Operand, TTIRBuilder

from helpers import *

seq_len = 128
num_of_heads = 32
head_dim = 128  # int(hidden_stage_size / num_of_heads)
hidden_stage_size = num_of_heads * head_dim
rotary_dim = int(head_dim / 2)

print(f"hidden_stage_size: {hidden_stage_size}")


@compile_to_flatbuffer(
    [
        (1, seq_len, hidden_stage_size),  # arg0 # (1, 12, 3200)
        (1, 1, seq_len, seq_len),  # arg1 # (1, 1, 12, 12)
        (1, seq_len),  # arg2 # (1, 12)
        (1, rotary_dim, 1),  # arg3 # (1, 50, 1)
        (1, num_of_heads, rotary_dim, head_dim),  # arg4 # (1, 32, 50, 100)
        (1, 1),  # arg5 # (1, 1)
        (1, num_of_heads, rotary_dim, head_dim),  # arg6 # (1, 32, 50, 100)
        (1, num_of_heads, rotary_dim, head_dim),  # arg7 # (1, 32, 50, 100)
        (1, 1),  # arg8 # (1, 1)
        (1, num_of_heads, rotary_dim, head_dim),  # arg9 # (1, 32, 50, 100)
        (1, 1),  # arg10 # (1, 1)
        (hidden_stage_size, hidden_stage_size),  # arg11 # (3200, 3200)
        (hidden_stage_size, hidden_stage_size),  # arg12 # (3200, 3200)
        (hidden_stage_size, hidden_stage_size),  # arg13 # (3200, 3200)
        (hidden_stage_size, hidden_stage_size),  # arg14 # (3200, 3200)
    ],
    targets=["ttnn"],
    module_dump=True,
)
def test_shape_calc_llama_attention(
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
    arg13: Operand,
    arg14: Operand,
    builder: TTIRBuilder,
):

    print(f"        {tuple(builder._get_golden_tensor(arg0).shape)},  # arg0")
    print(f"        {tuple(builder._get_golden_tensor(arg1).shape)},  # arg1")
    print(f"        {tuple(builder._get_golden_tensor(arg2).shape)},  # arg2")
    print(f"        {tuple(builder._get_golden_tensor(arg3).shape)},  # arg3")
    print(f"        {tuple(builder._get_golden_tensor(arg4).shape)},  # arg4")
    print(f"        {tuple(builder._get_golden_tensor(arg5).shape)},  # arg5")
    print(f"        {tuple(builder._get_golden_tensor(arg6).shape)},  # arg6")
    print(f"        {tuple(builder._get_golden_tensor(arg7).shape)},  # arg7")
    print(f"        {tuple(builder._get_golden_tensor(arg8).shape)},  # arg8")
    print(f"        {tuple(builder._get_golden_tensor(arg9).shape)},  # arg9")
    print(f"        {tuple(builder._get_golden_tensor(arg10).shape)},  # arg10")
    print(f"        {tuple(builder._get_golden_tensor(arg11).shape)},  # arg11")
    print(f"        {tuple(builder._get_golden_tensor(arg12).shape)},  # arg12")
    print(f"        {tuple(builder._get_golden_tensor(arg13).shape)},  # arg13")
    print(f"        {tuple(builder._get_golden_tensor(arg14).shape)},  # arg14")

    output1 = builder.squeeze(arg0, 0)
    output3 = builder.matmul(output1, arg11)
    output5 = builder.reshape(output3, (1, seq_len, num_of_heads, head_dim))
    output7 = builder.transpose(output5, -3, -2)
    output9 = builder.unsqueeze(arg2, 1)
    output11 = builder.matmul(arg3, output9)
    output13 = builder.transpose(output11, -2, -1)
    output15 = builder.concat([output13, output13], -1)
    output17 = builder.cos(output15)
    output19 = builder.unsqueeze(output17, 1)
    output21 = builder.multiply(output7, output19)
    output23 = builder.transpose(output7, -2, -1)
    output25 = builder.matmul(arg4, output23)
    output27 = builder.transpose(output25, -2, -1)
    output29 = builder.multiply(output27, arg5)
    output31 = builder.transpose(output7, -2, -1)
    output33 = builder.matmul(arg6, output31)
    output35 = builder.transpose(output33, -2, -1)
    output37 = builder.concat([output29, output35], -1)
    output39 = builder.sin(output15)
    output41 = builder.unsqueeze(output39, 1)
    output43 = builder.multiply(output37, output41)
    output45 = builder.add(output21, output43)
    output47 = builder.squeeze(output45, 0)
    output49 = builder.matmul(output1, arg12)
    output51 = builder.reshape(output49, (1, seq_len, num_of_heads, head_dim))
    output53 = builder.transpose(output51, -3, -2)
    output55 = builder.multiply(output53, output19)
    output57 = builder.transpose(output53, -2, -1)
    output59 = builder.matmul(arg7, output57)
    output61 = builder.transpose(output59, -2, -1)
    output63 = builder.multiply(output61, arg8)
    output65 = builder.transpose(output53, -2, -1)
    output67 = builder.matmul(arg9, output65)
    output69 = builder.transpose(output67, -2, -1)
    output71 = builder.concat([output63, output69], -1)
    output73 = builder.multiply(output71, output41)
    output75 = builder.add(output55, output73)
    output77 = builder.squeeze(output75, 0)
    output79 = builder.transpose(output77, -2, -1)
    output81 = builder.matmul(output47, output79)
    output83 = builder.unsqueeze(output81, 0)
    output85 = builder.multiply(output83, arg10)
    output87 = builder.add(output85, arg1)
    output89 = builder.softmax(output87, -1)
    output91 = builder.squeeze(output89, 0)
    output93 = builder.matmul(output1, arg13)
    output95 = builder.reshape(output93, (1, seq_len, num_of_heads, head_dim))
    output97 = builder.transpose(output95, -3, -2)
    output99 = builder.transpose(output97, -2, -1)
    output101 = builder.squeeze(output99, 0)
    output103 = builder.transpose(output101, -2, -1)
    output105 = builder.matmul(output91, output103)
    output107 = builder.unsqueeze(output105, 0)
    output109 = builder.transpose(output107, -3, -2)
    output111 = builder.reshape(output109, (seq_len, hidden_stage_size))
    output113 = builder.matmul(output111, arg14)
    output115 = builder.unsqueeze(output113, 0)

    return output115


if __name__ == "__main__":
    test_shape_calc_llama_attention()
