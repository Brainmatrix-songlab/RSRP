from typing import Any, Tuple, Optional

import jax
import jax.numpy as jnp

import flax
import optax

from recurrent_pg.structures import OnPolicyTrajectories
from recurrent_pg.utils.ema import EMA, EMAState


@flax.struct.dataclass
class PPOPreprocessorState:
    ema_obs: EMAState
    ema_rew: EMAState


@flax.struct.dataclass
class PPO:
    # RL & Advantage
    gamma:        float = 0.99
    gae_lambda:   float = 0.95

    # PPO
    eps:          float = 0.2
    vf_coeff:     float = 1.0
    entropy_coef: float = 1e-3

    # Optimization
    lr:           float        = 3e-4
    grad_norm: Optional[float] = 0.5

    # Env processing
    normalize_obs:    bool = True
    normalize_rew:    bool = True
    normalize_alpha:  float = 0.9999

    # Loss computation
    def _calculate_targets(self, trajectories: OnPolicyTrajectories) -> Tuple[jax.Array, jax.Array]:
        # Trajectories: [T, BSZ, ...]

        # FIXME: Unable to calculate value of last step, ignoring it
        #        Setting adv to zero makes the gradient zero for both value and policy
        next_values = trajectories.policy_output["value"][1:]

        rews        = trajectories.rew[:-1]
        dones       = trajectories.done[:-1]
        values      = trajectories.policy_output["value"][:-1]

        # GAE targets
        # Ref (rlax truncated_generalized_advantage_estimation): https://github.com/deepmind/rlax/blob/master/rlax/_src/multistep.py
        discounts   = self.gamma * (1 - dones)
        deltas      = rews + discounts * next_values - values

        def _gae_step(acc, inputs):
            delta_t, discount_t = inputs
            acc = delta_t + self.gae_lambda * discount_t * acc

            return acc, acc

        _, target_advs = jax.lax.scan(_gae_step, jnp.zeros_like(deltas[0]), (deltas, discounts), reverse=True)

        # Pad last adv to zero, see FIXME
        target_advs    = jax.lax.pad(target_advs, 0., ((0, 1, 0),) + ((0, 0, 0),) * (target_advs.ndim - 1))
        target_values  = target_advs + trajectories.policy_output["value"]

        return target_advs, target_values
    
    def loss_fn(self, key: Any, policy_output: Any, trajectories: OnPolicyTrajectories) -> Tuple[jax.Array, dict]:
        # Calculate targets
        target_advs, target_values = self._calculate_targets(trajectories)

        # PPO loss fn
        log_prob  = policy_output["act_dist"].log_prob(trajectories.policy_output["sampled_act"])
        log_ratio = log_prob - trajectories.policy_output["sampled_act_logprob"]
        ratio     = jnp.exp(log_ratio)

        surr1     = target_advs * ratio
        surr2     = target_advs * jnp.clip(ratio, 1.0 - self.eps, 1.0 + self.eps)

        value_err = policy_output["value"] - target_values

        # Losses
        clip_loss = -jnp.mean(jnp.minimum(surr1, surr2))
        vf_loss   =  jnp.mean(jax.lax.square(value_err))
        ent_loss  = -jnp.mean(policy_output["act_dist"].entropy(seed=key))

        loss      = clip_loss + self.vf_coeff * vf_loss + self.entropy_coef * ent_loss

        # Metrics
        approxkl = jnp.mean(ratio - 1.0 - log_ratio)
        ev       = 1 - jnp.var(value_err) / jnp.var(target_values)

        return loss, {
            # Losses
            "clip_loss": clip_loss,
            "vf_loss":   vf_loss,
            "entropy":  -ent_loss,

            # Metrics
            "approxkl": approxkl,
            "ev":       ev
        }

    # Gradient update
    @property
    def _optimizer(self):
        # setup optimizer
        return optax.chain(optax.clip_by_global_norm(self.grad_norm) if self.grad_norm is not None else optax.identity(),
                           optax.adam(learning_rate=self.lr))

    def gradient_init(self, params: Any):
        return self._optimizer.init(params)

    def gradient_update(self, params: Any, opt_state: Any, grads: Any) -> Tuple[Any, Any]:
        updates, new_opt_state = self._optimizer.update(grads, opt_state, params)
        new_params             = optax.apply_updates(params, updates)

        return new_params, new_opt_state

    # Preprocessing
    @property
    def _ema(self):
        return EMA(alpha=self.normalize_alpha, mean_axis=0)

    def preprocess_init(self, obs: Any, rew: jax.Array):
        return PPOPreprocessorState(
            ema_obs=self._ema.init(obs) if self.normalize_obs else None,
            ema_rew=self._ema.init(rew) if self.normalize_rew else None,
        )

    def preprocess_obs(self, prep_carry: PPOPreprocessorState, obs: Any) -> Tuple[PPOPreprocessorState, Any]:
        if not self.normalize_obs:
            return prep_carry, obs
        
        new_ema_obs = self._ema.update(prep_carry.ema_obs, obs)
        norm_obs    = self._ema.normalize(new_ema_obs,     obs)
        return prep_carry.replace(ema_obs=new_ema_obs), norm_obs
    
    def preprocess_rew(self, prep_carry: PPOPreprocessorState, rew: jax.Array) -> Tuple[PPOPreprocessorState, jax.Array]:
        if not self.normalize_rew:
            return prep_carry, rew

        new_ema_rew = self._ema.update(prep_carry.ema_rew, rew)
        norm_rew    = self._ema.normalize(new_ema_rew,     rew, zero_mean=False) * (1 - self.gamma)
        return prep_carry.replace(ema_rew=new_ema_rew), norm_rew
