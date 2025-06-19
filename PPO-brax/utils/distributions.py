import math

import jax
import jax.numpy as jnp

import flax


_half_log2pi = 0.5 * math.log(2 * math.pi)
_log2        = math.log(2.)


@flax.struct.dataclass
class NormalTanhSample:
    normal_sample: jax.Array

    @property
    def value(self):
        return jax.nn.tanh(self.normal_sample)


@flax.struct.dataclass
class NormalTanhDistribution:
    mu: jax.Array
    std: jax.Array

    def _tanh_log_det_jacob(self, normal_sample):
        return 2. * (_log2 - normal_sample - jax.nn.softplus(-2. * normal_sample))

    def sample(self, seed):
        eps             = jax.random.normal(seed, self.mu.shape, self.mu.dtype)
        normal_sample   = self.std * eps + self.mu

        return NormalTanhSample(normal_sample)

    def log_prob(self, sample: NormalTanhSample) -> jax.Array:
        log_unnormalized  = -0.5 * jnp.square((sample.normal_sample - self.mu) / self.std)
        log_normalization = _half_log2pi + jnp.log(self.std)

        return log_unnormalized - log_normalization - self._tanh_log_det_jacob(sample.normal_sample)

    def sample_and_logprob(self, seed):
        eps             = jax.random.normal(seed, self.mu.shape, self.mu.dtype)
        normal_sample   = self.std * eps + self.mu
        normal_log_prob = -0.5 * jnp.square(eps) - _half_log2pi - jnp.log(self.std)

        return NormalTanhSample(normal_sample), normal_log_prob - self._tanh_log_det_jacob(normal_sample)

    def entropy(self, seed: jax.Array) -> jax.Array:
        log_normalization = _half_log2pi + jnp.log(self.std)
        normal_entropy    = 0.5 + log_normalization

        # NOTE: Sampling based approximated log_det_jacob
        return normal_entropy + self._tanh_log_det_jacob(self.sample(seed).normal_sample)
    
    def deterministic_sample(self) -> NormalTanhSample:
        # Return the sample with maximum probability
        return NormalTanhSample(self.mu)


@flax.struct.dataclass
class MultivariateDiagNormalTanhDistribution(NormalTanhDistribution):
    def sample_and_logprob(self, seed):
        sample, logprob = super().sample_and_logprob(seed)
        return sample, jnp.sum(logprob, axis=-1)
    
    def log_prob(self, sample: NormalTanhSample) -> jax.Array:
        logprob = super().log_prob(sample)
        return jnp.sum(logprob, axis=-1)

    def entropy(self, seed: jax.Array) -> jax.Array:
        entropy = super().entropy(seed)
        return jnp.sum(entropy, axis=-1)
