# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import ttrt
import platform
from functools import reduce, wraps
import operator
from typing import List, Tuple

ALL_BACKENDS = set(["ttnn", "ttmetal", "ttnn-standalone"])
ALL_SYSTEMS = set(["n150", "n300", "llmbox", "tg", "p150", "p300"])


def is_x86_machine():
    machine = platform.machine().lower()
    return machine in ["x86_64", "amd64", "i386", "i686", "x86"]


x86_only = pytest.mark.skipif(
    not is_x86_machine(),
    reason=f"Test requires x86 architecture, but running on {platform.machine()}",
)


def pytest_addoption(parser):
    parser.addoption(
        "--path",
        action="store",
        default=".",
        help="Path to store test artifacts (e.g. flatbuffers and .mlir files)",
    )
    parser.addoption(
        "--sys-desc",
        action="store",
        default="ttrt-artifacts/system_desc.ttsys",
        help="Path to system descriptor",
    )
    parser.addoption(
        "--require-exact-mesh",
        action="store_true",
        help="Require exact mesh shape match with the current device (default allows subset)",
    )


def get_board_id(system_desc) -> str:
    arch = system_desc["chip_descs"][0]["arch"]
    num_chips = len(system_desc["chip_desc_indices"])

    match arch, num_chips:
        case "Blackhole", 1:
            return "p150"
        case "Blackhole", 2:
            return "p300"
        case "Wormhole_b0", 1:
            return "n150"
        case "Wormhole_b0", 2:
            return "n300"
        case _:
            raise ValueError(f"Unknown architecture: {arch}")


def filter_valid_mesh_shape(system_desc, params, require_exact_mesh=False):
    num_chips = reduce(operator.mul, params.get("mesh_shape", [1]), 1)
    num_physical_chips = len(system_desc["chip_desc_indices"])
    if require_exact_mesh:
        return num_chips == num_physical_chips
    else:
        return num_chips <= num_physical_chips


def pytest_collection_modifyitems(config, items):
    valid_items = []
    deselected = []
    system_desc = ttrt.binary.fbb_as_dict(
        ttrt.binary.load_system_desc_from_path(config.option.sys_desc)
    )["system_desc"]

    for item in items:
        # Only check parameterized tests
        if hasattr(item, "callspec"):
            params = item.callspec.params
            if not filter_valid_mesh_shape(
                system_desc, params, require_exact_mesh=config.option.require_exact_mesh
            ):
                # Deselect the test case
                deselected.append(item)
                continue
        valid_items.append(item)

        # Skip specific target / system combinations

        # Fetch the current target of this test, if any
        current_target = None
        for param in item.callspec.params.items():
            if param[0] == "target":
                current_target = param[1]
                break

        for marker in item.iter_markers(name="skip_config"):
            for platform_config in marker.args:

                # All of the operations we need to do on these are set membership based
                platform_config = set(platform_config)

                reason = marker.kwargs.get("reason", "")

                # Verify this is a valid configuration
                if not platform_config <= ALL_BACKENDS.union(ALL_SYSTEMS):
                    outliers = platform_config - ALL_BACKENDS.union(ALL_SYSTEMS)
                    raise ValueError(
                        f"Invalid skip config: {platform_config}, invalid entries: {outliers}. Please ensure that all entries in the config are members of {ALL_SYSTEMS} or {ALL_BACKENDS}"
                    )

                board_id = get_board_id(system_desc)

                if platform_config <= set([current_target, board_id]):
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"Operation not supported on following platform/target combination: {platform_config}. {reason}"
                        )
                    )

    # Update the items list (collected tests)
    items[:] = valid_items

    # Report deselected items to pytest
    if deselected:
        config.hook.pytest_deselected(items=deselected)


# ------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------
def _calc_shard_args(
    mesh_shape: Tuple[int, ...], test_shape: Tuple[int]
) -> Tuple[List[int], List[int], List[int]]:
    """
    Derive shard_shape / shard_dims and expand input_shape so that
    each sharded dim is multiplied by its mesh factor.
    """
    rank_in = len(test_shape)
    rank_mesh = len(mesh_shape)

    # Take the last `rank_mesh` dims as sharded dims
    shard_dims = list(range(rank_in - rank_mesh, rank_in))
    shard_shape = [1] * rank_in
    for d, factor in zip(shard_dims, mesh_shape):
        shard_shape[d] = factor

    full_input_shape = list(test_shape)
    for d, factor in zip(shard_dims, mesh_shape):
        full_input_shape[d] *= factor

    return shard_shape, shard_dims, full_input_shape


# ------------------------------------------------------------
# Fixture
# ------------------------------------------------------------
@pytest.fixture
def shard_wrap_factory(test_shape, mesh_shape):
    """
    Factory fixture:
        mesh_wrap_factory(fn) -> (input_shape, wrapped_fn)

    wrapped_fn sharding, calls `fn`, then unsharding.
    """
    shard_shape, shard_dims, full_input_shape = _calc_shard_args(mesh_shape, test_shape)

    def _factory(test_fn):
        @wraps(test_fn)  # keep original name for debugging
        def wrapped_fn(in0, builder):
            # sharding
            in_shard = builder.mesh_shard(
                in0,
                shard_direction="#ttcore.shard_direction<full_to_shard>",
                shard_type="#ttcore.shard_type<devices>",
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )
            # op under test
            out_shard = test_fn(in_shard, builder)
            # unsharding
            return builder.mesh_shard(
                out_shard,
                shard_direction="#ttcore.shard_direction<shard_to_full>",
                shard_type="#ttcore.shard_type<devices>",
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

        return full_input_shape, wrapped_fn

    return _factory
