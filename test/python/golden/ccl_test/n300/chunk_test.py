# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch


a = torch.randn([1, 32, 128, 128])
b = torch.randn([1, 32, 128, 128])
a_chunks = torch.chunk(a, 2, dim=2)
b_chunks = torch.chunk(a, 2, dim=2)

golden = torch.concat((a, b), dim=3)
concat_0 = torch.concat((a_chunks[0], b_chunks[0]), dim=2)
concat_1 = torch.concat((a_chunks[1], b_chunks[1]), dim=2)
output = torch.concat((concat_0, concat_1), dim=3)
print(f"concat check result : {torch.allclose(golden, output)}")


golden = torch.transpose(a, -1, -2)
transpose_0 = torch.transpose(a_chunks[0], -1, -2)
transpose_1 = torch.transpose(a_chunks[1], -1, -2)
output = torch.concat((transpose_0, transpose_1), dim=3)
print(f"transpose check result : {torch.allclose(golden, output)}")
