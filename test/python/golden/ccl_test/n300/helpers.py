# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Helper: Golden Management
class _GS:
    # GoldenStore
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_GS, cls).__new__(cls)
            cls._instance.output_goldens = []
            cls._instance.input_goldens = []
        return cls._instance

    def add_output_golden(self, golden):
        self.output_goldens.append(golden)

    def add_input_golden(self, golden):
        self.input_goldens.append(golden)

    def clear_input_goldens(self):
        self.input_goldens = []

    def clear_output_goldens(self):
        self.output_goldens = []

    def get_input_goldens(self):
        return self.input_goldens

    def get_output_goldens(self):
        return self.output_goldens

    def change_input_golden(self, idx, golden):
        self.input_goldens[idx] = golden


def save_all_input_goldens(args: list, builder):
    _GS().clear_input_goldens()
    for arg in args:
        _GS().add_input_golden(builder._get_golden_tensor(arg))


def save_all_output_goldens(args: list, builder):
    _GS().clear_output_goldens()
    for arg in args:
        _GS().add_output_golden(builder._get_golden_tensor(arg))


def change_input_golden(idx, golden):
    _GS().change_input_golden(idx, golden)


def set_graph_goldens(builder):
    builder.set_graph_input_output(
        _GS().get_input_goldens(), _GS().get_output_goldens()
    )


# Helper: Mesh Sharding


def full_to_shard_device(input, builder, dim):
    rank = len(builder._get_golden_tensor(input).shape)
    shard_shape = [1] * rank
    shard_shape[dim] = 2
    return builder.mesh_shard(
        input,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=shard_shape,
        shard_dims=[-1, dim],
    )


def full_to_shard_replicate(input, builder):
    return builder.mesh_shard(
        input,
        shard_direction="#tt.shard_direction<full_to_shard>",
        shard_type="#tt.shard_type<replicate>",
        shard_shape=[1],
        shard_dims=[-1],
    )


def shard_to_full_device(input, builder, dim):
    rank = len(builder._get_golden_tensor(input).shape)
    shard_shape = [1] * rank
    shard_shape[dim] = 2
    return builder.mesh_shard(
        input,
        shard_direction="#tt.shard_direction<shard_to_full>",
        shard_type="#tt.shard_type<devices>",
        shard_shape=shard_shape,
        shard_dims=[-1, dim],
    )


def shard_to_full_replicate(input, builder):
    return builder.mesh_shard(
        input,
        shard_direction="#tt.shard_direction<shard_to_full>",
        shard_type="#tt.shard_type<replicate>",
        shard_shape=[1],
        shard_dims=[-1],
    )
