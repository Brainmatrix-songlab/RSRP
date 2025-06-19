from typing import Any

import jax
import jax.numpy as jnp

import flax


@flax.struct.dataclass
class MetricAggregatorState:
    sum:   Any
    count: jax.Array


class MetricAggregator:
    @staticmethod
    def init(tree: Any) -> MetricAggregatorState:
        sum   = jax.tree_map(lambda x: jnp.zeros_like(x), tree)
        count = jnp.zeros((), jnp.int32)

        return MetricAggregatorState(sum=sum, count=count)

    @staticmethod
    def push(state: MetricAggregatorState, tree: Any, count: int = 1):
        new_sum   = jax.tree_map(lambda s, x: s + x, state.sum, tree)
        new_count = state.count + count

        return MetricAggregatorState(sum=new_sum, count=new_count)

    @staticmethod
    def pop(state: MetricAggregatorState):
        return MetricAggregator.init(state.sum), \
               jax.tree_map(lambda x: x / state.count, state.sum)
