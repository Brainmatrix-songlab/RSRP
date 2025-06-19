import jax
import jax.numpy as jnp

from brax import envs


class BraxVector:
    def __init__(self,
                 name: str,
                 max_episode_length: int = 1000,
                 action_repeat: int = 1,
                 *args, **kwargs):
        self.max_episode_length = max_episode_length

        # Create brax env
        self.env = envs.get_environment(name, *args, **kwargs)
        self.env = envs.wrappers.EpisodeWrapper(self.env, max_episode_length, action_repeat)
        self.env = envs.wrappers.VmapWrapper(self.env)

    def reset(self, key: jax.Array) -> envs.State:
        return self.env.reset(key)

    def step(self, state: envs.State, act: jax.Array) -> envs.State:
        state = self.env.step(state, act)
        return state, state.reward, state.done.astype(jnp.bool_)

    def observe(self, state: envs.State):
        return state.obs

    @property
    def observation_size(self) -> int:
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size
