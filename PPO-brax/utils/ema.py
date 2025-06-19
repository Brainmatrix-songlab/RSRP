from typing import Any

import jax
import jax.numpy as jnp

import flax


def lerp(y, x, alpha):
    # Linear interpolation
    # = alpha * y + (1 - alpha) * x
    return x + alpha * (y - x)


@flax.struct.dataclass
class EMAState:
    mu: Any
    nu: Any

    decay_product: jax.Array


@flax.struct.dataclass
class EMA:
    alpha:     float
    mean_axis: int = 0

    eps:       float = 1e-5

    def init(self, batch: Any) -> EMAState:
        init_zeros = jax.tree_map(lambda x: jnp.zeros_like(jnp.mean(x)), batch)
        init_state = EMAState(mu=init_zeros, nu=init_zeros, decay_product=1.0)

        return self.update(init_state, batch)

    def update(self, state: EMAState, batch: Any) -> EMAState:
        new_mu = jax.tree_map(lambda mu, x: lerp(mu, jnp.mean(x,             self.mean_axis), self.alpha), state.mu, batch)
        new_nu = jax.tree_map(lambda nu, x: lerp(nu, jnp.mean(jnp.square(x), self.mean_axis), self.alpha), state.nu, batch)
        new_decay_product = state.decay_product * self.alpha

        return EMAState(mu=new_mu, nu=new_nu, decay_product=new_decay_product)
    
    def normalize(self, state: EMAState, batch: Any, zero_mean=True, unit_variance=True) -> Any:
        debias = 1 / (1 - state.decay_product)

        def _normalize_x(x, mu, nu):
            mean = mu * debias
            var  = jnp.maximum(nu * debias - jnp.square(mean), 0)

            if zero_mean:
                x = x - mean
            if unit_variance:
                x = x * jax.lax.rsqrt(var + self.eps)
            return x

        return jax.tree_map(_normalize_x, batch, state.mu, state.nu)
