# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: SYSTEM_DESC_PATH=%system_desc_path% %python %s

import inspect

from ttmlir.test_utils import compile_to_flatbuffer, set_output_path
from ttmlir.ttir_builder import Operand, TTIRBuilder


@compile_to_flatbuffer(
    [
        (128, 128),  # arg0
    ],
    targets=["ttnn"],
    module_dump=True,
)
def test_multi_output(
    arg0: Operand,
    builder: TTIRBuilder,
):
    output_neg = builder.neg(arg0)
    output_cos = builder.cos(output_neg)
    output_cos = builder.reshape(output_cos, [1, 4, 32, 128])
    output_sin = builder.sin(output_neg)
    output_sin = builder.reshape(output_sin, [1, 4, 128, 32])
    return output_cos


if __name__ == "__main__":
    test_multi_output()
