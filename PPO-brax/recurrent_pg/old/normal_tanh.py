from typing import Any, Tuple

import jax
import jax.numpy as jnp

import flax
import flax.linen as nn
import distrax

from recurrent_pg.networks import NETWORKS


class NormalTanh(nn.Module):
    act_dims:     int

    network_type: str
    network_conf: dict = None

    @nn.compact
    def __call__(self, carry, obs) -> Tuple[Any, jax.Array]:
        # Network and logstd
        network = NETWORKS[self.network_type](out_dims=self.act_dims + 1, **(self.network_conf or {}))
        log_std = self.param("log_std", nn.initializers.zeros, (self.act_dims, ))

        # Distribution
        new_carry, output = network(carry, obs)
        value, mu         = output[..., 0], output[..., 1:]
        std               = jnp.exp(log_std)

        act_dist  = distrax.Transformed(distrax.MultivariateNormalDiag(loc=mu, scale_diag=std), distrax.Tanh())
        return new_carry, (act_dist, value)
    
    def initial_carry(self, batch_size: int) -> Any:
        return NETWORKS[self.network_type].initial_carry(batch_size, self.network_conf or {})
