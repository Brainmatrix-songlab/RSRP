from typing import Optional, Callable
from functools import partial
import math

import jax
import jax.numpy as jnp
import flax.linen as nn

from recurrent_pg.utils.functions import get_index, get_segment
from recurrent_pg.utils.distributions import MultivariateDiagNormalTanhDistribution


# Surrogate functions
# max(0, 1 - alpha * abs(x))
@jax.custom_jvp
def heaviside_surr_abs(x, beta):
    return (x > 0.0).astype(x.dtype)


@heaviside_surr_abs.defjvp
def heaviside_surr_abs_jvp(primals, tangents):
    x, beta = primals
    grad_dot = tangents[0]

    primal_out = heaviside_surr_abs(x, beta)

    phi = jnp.maximum(1 - beta * jnp.abs(x), 0)
    tangent_out = grad_dot * phi
    return primal_out, tangent_out


# SuperSpike, 1 / (beta * |x| + 1)^2
@jax.custom_jvp
def heaviside_surr_superspike(x, beta):
    return (x > 0.0).astype(x.dtype)


@heaviside_surr_superspike.defjvp
def heaviside_surr_superspike_jvp(primals, tangents):
    x, beta = primals
    grad_dot = tangents[0]

    primal_out = heaviside_surr_superspike(x, beta)

    tangent_out = grad_dot / jax.lax.square(beta * jax.lax.abs(x) + 1)
    return primal_out, tangent_out


# Sigmoid', 
@jax.custom_jvp
def heaviside_surr_dsigmoid(x, beta):
    return (x > 0.0).astype(x.dtype)


@heaviside_surr_dsigmoid.defjvp
def heaviside_surr_dsigmoid_jvp(primals, tangents):
    x, beta = primals
    grad_dot = tangents[0]

    primal_out = heaviside_surr_dsigmoid(x, beta)

    exp_betax = jax.lax.exp(beta * x)
    tangent_out = grad_dot * beta * exp_betax / jax.lax.square(exp_betax + 1)
    return primal_out, tangent_out


SURROGATE_FUNCTIONS = {
    "heaviside_surr_abs": heaviside_surr_abs,
    "heaviside_surr_superspike": heaviside_surr_superspike,
    "heaviside_surr_dsigmoid": heaviside_surr_dsigmoid,
}


# Utility functions
def lerp(y, x, alpha):
    # Linear interpolation
    # = alpha * y + (1 - alpha) * x
    return x + alpha * (y - x)


def conn_dense(kernel, x):
    # matmul
    return jax.lax.dot_general(x, kernel, (((x.ndim - 1,), (0,)), ((), ())))


class SG_RSNN(nn.Module):
    """Recurrent spiking neural network with LIF model

    Same architecture and parameters as conn_snn, except using real weights and surrogate spike gradient."""

    # Network parameters
    out_dims: int

    num_neurons: int = 256

    # rand_init_Vm: bool = True

    dtype: jnp.dtype = jnp.float32

    # SG parameters
    surrogate_function: str = "heaviside_surr_superspike"
    surrogate_beta:   float = 20.0

    # SNN simulation
    sim_time: float = 16.6  # ms
    dt: float       = 0.5   # ms

    # SNN parameters
    K_in:  float = 0.1
    K_h:   float = 1.0
    K_out: float = 5.0

    tau_syn:  float = 5.0   # ms
    tau_Vm:   float = 10.0  # ms
    tau_out:  float = 10.0  # ms

    Vth:      float = 1.0

    @nn.compact
    def __call__(self, carry, x):
        # Kernels
        in_dims        = x.shape[-1]

        kernel_in  = self.param("kernel_in",  nn.initializers.normal(stddev=1.0), (2 * in_dims, self.num_neurons),      self.dtype)
        kernel_h   = self.param("kernel_h",   nn.initializers.normal(stddev=1.0), (self.num_neurons, self.num_neurons), self.dtype)
        kernel_out = self.param("kernel_out", nn.initializers.normal(stddev=1.0), (self.num_neurons, self.out_dims),    self.dtype)

        # Parameters
        R_in  = self.K_in  * self.Vth * self.tau_Vm                * math.sqrt(2 / in_dims)
        R     = self.K_h   * self.Vth * self.tau_Vm / self.tau_syn * math.sqrt(2 / self.num_neurons)
        R_out = self.K_out                                         * math.sqrt(1 / self.num_neurons)

        alpha_syn = math.exp(-self.dt / self.tau_syn)
        alpha_Vm  = math.exp(-self.dt / self.tau_Vm)
        alpha_out = math.exp(-self.dt / self.tau_out)

        # input layer
        x    = x.astype(self.dtype)
        i_in = R_in * conn_dense(kernel_in, jnp.concatenate([x, -x], axis=-1))

        # SNN layer
        def _snn_step(_carry, _x):
            v_m, i_syn, rate, spike = _carry

            i_spike = R * conn_dense(kernel_h, spike)
            i_syn   = i_syn * alpha_syn + i_spike
            v_m     = lerp(v_m, i_syn + i_in, alpha_Vm)

            spike   = SURROGATE_FUNCTIONS[self.surrogate_function](v_m - self.Vth, self.surrogate_beta)
            v_m     = (1 - spike) * v_m

            rate    = lerp(rate, (1 / self.dt) * spike, alpha_out)

            return (v_m, i_syn, rate, spike), None

        def _snn_get_output(_carry):
            v_m, i_syn, rate, spike = _carry

            return R_out * conn_dense(kernel_out, rate)

        # Stepping
        carry, _ = jax.lax.scan(_snn_step, carry, None, round(self.sim_time / self.dt))
        return carry, _snn_get_output(carry)

    @staticmethod
    def initial_carry(batch_size: int, num_neurons: int = 256, dtype: jnp.dtype = jnp.float32, **kwargs):
        v_m   = jnp.zeros((batch_size, num_neurons), dtype)
        i_syn = jnp.zeros((batch_size, num_neurons), dtype)
        rate  = jnp.zeros((batch_size, num_neurons), dtype)
        spike = jnp.zeros((batch_size, num_neurons), dtype)

        # if self.rand_init_Vm:
        #     # Random init Vm to [Vr, Vth]
        #     v_m = jax.random.uniform(key, (batch_size, self.num_neurons), self.dtype, 0, self.Vth)

        return v_m, i_syn, rate, spike

    @staticmethod
    def carry_metrics(carry):
        v_m, i_syn, rate, spike = carry

        return {
            "spikes_per_ms": jnp.mean(jnp.abs(rate)),
            "avg_i_syn":     jnp.mean(jnp.abs(i_syn))
        }


class RSNN(nn.Module):
    act_dims: int
    new_episode: str = "zero"

    min_std: float = 1e-3

    snn_config: Optional[dict] = None

    bptt_truncate: Optional[int] = None

    # DNN vf
    vf_use_lstm:  bool = False
    vf_lstm_units: int = 256

    def setup(self):
        self.backbone  = SG_RSNN(self.act_dims + 1, **(self.snn_config or {}))

        self.std       = self.param("std", nn.initializers.ones, (self.act_dims, ))

        if self.vf_use_lstm:
            self.vf_backbone = nn.OptimizedLSTMCell()
            self.vf_head     = nn.Dense(1)

    def infer(self, carry, inputs, is_new_episode, key=None, is_eval=False):
        # New episode processing
        if self.new_episode == "zero":
            def _zero_hidden(x):
                new_ep = is_new_episode.reshape([-1] + [1] * (x.ndim - 1))
                return jnp.where(new_ep, 0, x)

            carry = jax.tree_map(_zero_hidden, carry)
        elif self.new_episode == "flag":
            inputs = jnp.concatenate([jnp.expand_dims(is_new_episode, axis=-1).astype(inputs.dtype), inputs], axis=-1)
        elif self.new_episode == "none":
            pass
        else:
            raise NotImplementedError(f"Unknown new episode processing: {self.new_episode}")

        # RSNN
        new_snn_carry, output = self.backbone(carry["snn"], inputs)
        value, mu         = output[..., 0], output[..., 1:]

        # DNN Vf
        new_lstm_carry = None
        if self.vf_use_lstm:
            new_lstm_carry, vf_output = self.vf_backbone(carry["lstm"], inputs)
            value                     = jnp.squeeze(self.vf_head(vf_output), axis=-1)

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

        return {"snn": new_snn_carry, "lstm": new_lstm_carry}, (act, policy_output)

    def __call__(self, carry, inputs, is_new_episode):
        infer_seq = nn.transforms.scan(RSNN.infer,
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
        snn_carry  = SG_RSNN.initial_carry(batch_size, **(self.snn_config or {}))

        lstm_carry = None
        if self.vf_use_lstm:
            lstm_carry = (jnp.zeros((batch_size, self.vf_lstm_units)), jnp.zeros((batch_size, self.vf_lstm_units)))

        return {"snn": snn_carry, "lstm": lstm_carry}
