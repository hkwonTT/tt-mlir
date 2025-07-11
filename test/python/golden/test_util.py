# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch, math, itertools
from typing import List, Tuple, Iterable, Sequence, Union


def all_sharding_configs_2d(
    input_rank: int, num_devices: int  # e.g. 4-D tensor -> 4  # mesh size (rows*cols)
) -> Iterable[
    Tuple[
        Tuple[int, int],  # mesh_shape
        Tuple[int, int],  # shard_dims
        List[int],  # shard_shape
        int,  # cluster_axis
        List[List[int]],
    ]
]:
    # all 2-D mesh factorizations (row, col) – both orientations
    mesh_shapes = {
        (r, num_devices // r)
        for r in range(1, math.isqrt(num_devices) + 1)
        if num_devices % r == 0
    }
    mesh_shapes |= {(c, r) for r, c in mesh_shapes}

    # every pair of shard dims (-1 = not sharded)
    shard_dim_pairs = itertools.product(range(-1, input_rank), repeat=2)

    for mesh_shape, shard_dims in itertools.product(
        sorted(mesh_shapes), shard_dim_pairs
    ):
        dim_row, dim_col = shard_dims
        shard_shape = [
            mesh_shape[0] if i == dim_row else mesh_shape[1] if i == dim_col else 1
            for i in range(input_rank)
        ]

        # — skip invalids: duplicate dim or “no sharding at all”
        if dim_row == dim_col or all(s == 1 for s in shard_shape):
            continue

        rows, cols = mesh_shape
        for cluster_axis in (0, 1):
            replica_groups = (
                [
                    [r * cols + c for r in range(rows)] for c in range(cols)
                ]  # across rows
                if cluster_axis == 0
                else [
                    [r * cols + c for c in range(cols)] for r in range(rows)
                ]  # across cols
            )
            yield mesh_shape, shard_dims, shard_shape, cluster_axis, replica_groups


def shape_divisible(tensor_shape: Sequence[int], shard_shape: Sequence[int]) -> bool:
    return all(t % s == 0 for t, s in zip(tensor_shape, shard_shape))


def shardTensor2dMesh(
    tensor: torch.Tensor, mesh_shape: Tuple[int, int], shard_dims: Tuple[int, int]
):
    rows, cols = mesh_shape
    row_dim, col_dim = shard_dims

    # Shard along rows
    row_tensors = (
        [tensor.clone() for _ in range(rows)]
        if row_dim == -1
        else torch.chunk(tensor, rows, dim=row_dim)
    )

    # Shard along columns
    if col_dim == -1:
        return [t.clone() for t in row_tensors for _ in range(cols)]
    tensor_shards = [
        tt for t in row_tensors for tt in torch.chunk(t, cols, dim=col_dim)
    ]
    return tensor_shards


def concatMesh2dToTensor(
    device_shards: List[torch.Tensor],
    mesh_shape: Tuple[int, int],
    shard_dims: Tuple[int, int],
):
    rows, cols = mesh_shape
    row_dim, col_dim = shard_dims

    # Reshape the list of shards into a 2D list representing the device mesh
    mesh_shape = [
        device_shards[i : i + cols] for i in range(0, len(device_shards), cols)
    ]

    if col_dim == -1:
        row_concatenated = [row[0] for row in mesh_shape]
    else:
        row_concatenated = [torch.cat(row, dim=col_dim) for row in mesh_shape]

    # Then concatenate the resulting tensors along rows
    if row_dim == -1:
        return row_concatenated[0]
    else:
        return torch.cat(row_concatenated, dim=row_dim)


# ────────────────────────── utility helpers ──────────────────────────
def _factor_tuples(n: int, k: int) -> Iterable[Tuple[int, ...]]:
    """Yield every length-k tuple of positive ints whose product equals n."""
    if k == 1:
        yield (n,)
        return
    for f in range(1, int(math.isqrt(n)) + 1):
        if n % f == 0:
            for rest in _factor_tuples(n // f, k - 1):
                yield (f,) + rest


def _mesh_shapes(
    n_devices: int, min_dim: int, max_dim: int | None
) -> Iterable[Tuple[int, ...]]:
    """Ordered mesh shapes (tuples) with product n_devices and
    length in [min_dim, max_dim]."""
    max_dim = max_dim or n_devices  # None -> no upper bound
    for nd in range(min_dim, max_dim + 1):
        for base in _factor_tuples(n_devices, nd):
            # include every orientation
            yield from set(itertools.permutations(base))


def _ravel_nd(idx: Tuple[int, ...], shape: Tuple[int, ...]) -> int:
    """Convert N-D index to row-major flat index."""
    out, mult = 0, 1
    for size, i in zip(reversed(shape), reversed(idx)):
        out += i * mult
        mult *= size
    return out


def _replica_groups(shape: Tuple[int, ...], cluster_axis: int) -> List[List[int]]:
    """Groups that vary cluster_axis while fixing all other axes."""
    other_axes = [ax for ax in range(len(shape)) if ax != cluster_axis]
    groups: List[List[int]] = []
    for fixed in itertools.product(*[range(shape[ax]) for ax in other_axes]):
        group: List[int] = []
        for v in range(shape[cluster_axis]):
            full = list(fixed)
            full.insert(cluster_axis, v)
            group.append(_ravel_nd(tuple(full), shape))
        groups.append(group)
    return groups


def _is_valid(
    mesh_shape: Tuple[int, ...], shard_dims: Tuple[int, ...], input_rank: int
) -> tuple[bool, List[int]]:
    """Return (True, shard_shape) if configuration passes all checks."""
    # 1. Cannot shard along axis of size 1
    if any(ms == 1 and sd != -1 for ms, sd in zip(mesh_shape, shard_dims)):
        return False, []
    # 2. No duplicate tensor dimensions
    used = [d for d in shard_dims if d != -1]
    if len(set(used)) != len(used):
        return False, []
    # 3. At least one dimension must be sharded
    shard_shape = [1] * input_rank
    for axis, dim in enumerate(shard_dims):
        if dim != -1:
            shard_shape[dim] *= mesh_shape[axis]
    if all(s == 1 for s in shard_shape):
        return False, []
    return True, shard_shape


# ─────────────────────────────────────────────────────────────────────


def sweep_all_sharding_configs(
    input_rank: int,
    num_devices: int,
    *,
    min_mesh_dim: int = 1,
    max_mesh_dim: int | None = None,
) -> Iterable[
    Tuple[
        Tuple[int, ...],  # mesh_shape
        Tuple[int, ...],  # shard_dims (-1 = unsharded)
        List[int],  # shard_shape
        int,  # cluster_axis
        List[List[int]],  # replica_groups
    ]
]:
    """
    Generate every sharding configuration whose mesh dimensionality
    is in [min_mesh_dim, max_mesh_dim] and whose volume equals num_devices.

    Dimension-1 mesh axes are forced to remain unsharded.
    """
    if max_mesh_dim is not None and max_mesh_dim < min_mesh_dim:
        raise ValueError("max_mesh_dim must be >= min_mesh_dim")

    for mesh_shape in _mesh_shapes(num_devices, min_mesh_dim, max_mesh_dim):
        nd = len(mesh_shape)
        for shard_dims in itertools.product(range(-1, input_rank), repeat=nd):
            ok, shard_shape = _is_valid(mesh_shape, shard_dims, input_rank)
            if not ok:
                continue
            for cluster_axis in range(nd):
                yield (
                    mesh_shape,
                    shard_dims,
                    shard_shape,
                    cluster_axis,
                    _replica_groups(mesh_shape, cluster_axis),
                )


if __name__ == "__main__":
    all_configs = list(
        sweep_all_sharding_configs(4, 32, min_mesh_dim=3, max_mesh_dim=3)
    )
    for c in all_configs:
        mesh_shape, shard_dims, shard_shape, cluster_axis, replica_groups = c
        print(
            f"mesh_shape={mesh_shape}, shard_dims={shard_dims}, "
            f"shard_shape={shard_shape}, cluster_axis={cluster_axis}, "
            f"replica_groups={replica_groups}"
        )
