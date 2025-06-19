from typing import Optional
from functools import partial

import jax
import jax.numpy as jnp

import flax.linen as nn

from recurrent_pg.utils.functions import get_index, get_segment
from recurrent_pg.utils.distributions import MultivariateDiagNormalTanhDistribution


class LSTM(nn.Module):
    act_dims: int

    hidden_dims: int = 256
    new_episode: str = "zero"

    min_std: float = 1e-3

    bptt_truncate: Optional[int] = None

    def setup(self):
        self.backbone  = nn.OptimizedLSTMCell(features=256)
        self.act_head  = nn.Dense(self.act_dims)
        self.v_head    = nn.Dense(1)

        self.std       = self.param("log_std", nn.initializers.zeros, (self.act_dims, ))

    def infer(self, carry, inputs, is_new_episode, key=None, is_eval=False):
        # New episode processing
        if self.new_episode == "zero":
            def _zero_hidden(x):
                new_ep = is_new_episode.reshape([-1] + [1] * (x.ndim - 1))
                return jnp.where(new_ep, 0, x)

            carry = jax.tree_map(_zero_hidden, carry)
        elif self.new_episode == "flag":
            inputs = jnp.concatenate([is_new_episode.astype(inputs.dtype), inputs], axis=-1)
        elif self.new_episode == "none":
            pass
        else:
            raise NotImplementedError(f"Unknown new episode processing: {self.new_episode}")

        # LSTM backbone
        new_carry, feature = self.backbone(carry, inputs)
        # Output heads
        mu, value          = self.act_head(feature), jnp.squeeze(self.v_head(feature))
        std                = (jax.nn.softplus(self.std) + self.min_std).reshape((1, ) * (mu.ndim - 1) + (-1, ))
        act_dist           = MultivariateDiagNormalTanhDistribution(mu, std)

        # Policy output
        act           = None
        policy_output = dict(value=value, act_dist=act_dist)
        if key is not None:
            if is_eval:
                # Deterministic action
                act = act_dist.deterministic_sample().value
            else:
                # Sample action
                sampled_act, sampled_act_logprob = act_dist.sample_and_logprob(key)

                act = sampled_act.value
                policy_output.update(dict(
                    sampled_act=sampled_act,
                    sampled_act_logprob=sampled_act_logprob
                ))

        return new_carry, (act, policy_output)

    def __call__(self, carry, inputs, is_new_episode):
        infer_seq = nn.transforms.scan(LSTM.infer,
                                       variable_broadcast="params", split_rngs={"params": False})

        if self.bptt_truncate is None:
            # Full-length calculation
            return infer_seq(self, get_index(carry, 0),
                             inputs, is_new_episode)

        # Segmented calculation
        total_length = is_new_episode.shape[0]
        seg_length   = self.bptt_truncate
        assert total_length % seg_length == 0, f"Total length {total_length} must be a multiple of segment length {seg_length}."

        def _calc_segment(seg_index):
            seg_start = seg_index * seg_length

            return infer_seq(self, get_index(carry, seg_start),
                             get_segment(inputs, seg_start, seg_length), get_segment(is_new_episode, seg_start, seg_length))
        
        new_carry, outputs = jax.vmap(_calc_segment)(jnp.arange(total_length // seg_length))
        return get_index(new_carry, -1), jax.tree_map(partial(jnp.concatenate, axis=0), outputs)


    def initial_carry(self, batch_size):
        c = jnp.zeros((batch_size, self.hidden_dims))
        h = jnp.zeros((batch_size, self.hidden_dims))
        return c, h
