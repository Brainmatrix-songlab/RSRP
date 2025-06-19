from typing import Any

import jax


def get_index(x: Any, idx: int):
    return jax.tree_map(lambda e: e[idx], x)


def get_segment(x: Any, start: int, length: int):
    def _segment_slice(e):
        return jax.lax.dynamic_slice(e, (start, ) + (0, ) * (e.ndim - 1), (length, ) + e.shape[1:])

    return jax.tree_map(_segment_slice, x)
