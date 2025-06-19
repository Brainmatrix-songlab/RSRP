from typing import Any

import jax
import flax


@flax.struct.dataclass
class OnPolicyTrajectories:
    # Environment
    obs:  Any
    # act:  Any
    rew:  jax.Array
    done: jax.Array

    is_new_episode: jax.Array

    # Policy
    carry:         Any
    policy_output: Any
