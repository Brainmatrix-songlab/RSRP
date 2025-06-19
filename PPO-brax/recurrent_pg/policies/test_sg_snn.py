from typing import Any, Tuple
import time

import jax
import jax.numpy as jnp
import flax.linen as nn

from sg_rsnn import SG_RSNN


def test_sg_snn_speed(
    seed:        int = 0,
    test_rounds: int = 10,
    
    num_neurons: int = 256,

    batch_size: int = 1024,
    bptt_steps: int = 32,

    in_dims:    int = 240,
    out_dims:   int = 17,

    dtype: jnp.dtype = jnp.float32,
):
    key, key_net, key_carry = jax.random.split(jax.random.PRNGKey(seed), 3)

    # Create network class
    network_cls = nn.scan(SG_RSNN,
                          variable_broadcast="params",
                          split_rngs={"params": False},
                          in_axes=1, out_axes=1)
    
    network     = network_cls(out_dims=out_dims,
                             num_neurons=num_neurons,
                             dtype=dtype)

    # Initialize network
    carry = network.initial_carry(key_carry, batch_size)
    input_example = jnp.zeros((batch_size, bptt_steps, in_dims), dtype)

    params = jax.jit(network.init)(key_net, carry, input_example)

    # Test BP speed
    def _loss_fn(params, carry, xs, ys):
        new_carry, ys_pred = network.apply(params, carry, xs)

        loss = jnp.mean((ys - ys_pred) ** 2)
        return loss
    
    _jit_grad_loss_fn = jax.jit(jax.grad(_loss_fn))

    start_time = time.time()
    for _ in range(test_rounds):
        key, xs_key, ys_key = jax.random.split(key, 3)
        xs = jax.random.normal(xs_key, (batch_size, bptt_steps, in_dims))
        ys = jax.random.normal(ys_key, (batch_size, bptt_steps, out_dims))

        grad = _jit_grad_loss_fn(params, carry, xs, ys)
        jax.tree_map(lambda x: x.block_until_ready(), grad)

    print(f"Elapsed: {(time.time() - start_time) / test_rounds:.2f}")


if __name__ == "__main__":
    test_sg_snn_speed()
