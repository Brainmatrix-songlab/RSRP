from typing import Any, Tuple
from functools import partial
import time

import jax
import jax.numpy as jnp
import flax

from omegaconf import OmegaConf
from tqdm import tqdm
import wandb

from recurrent_pg.structures import OnPolicyTrajectories
from recurrent_pg.utils.metric_aggregator import MetricAggregator

from recurrent_pg.environments import ENVIRONMENTS
from recurrent_pg.algorithms import ALGORITHMS
from recurrent_pg.policies import POLICIES


@flax.struct.dataclass
class TrainConfig:
    # Max iteration
    max_iteration: int = 100000

    # Data collection
    batch_size:    int = 2048
    collect_len:   int = 16

    # Metrics
    metric_log_every:  int = 100
    metric_eval_every: int = 1000


@flax.struct.dataclass
class AgentConfig:
    algo_cls:   Any
    env_cls:    Any
    policy_cls: Any

    train:      TrainConfig


@flax.struct.dataclass
class AgentState:
    key: Any

    # Env
    env_states:         Any
    env_reset_pool:     Any
    env_is_new_episode: jax.Array

    # Policy
    policy_carry:     Any

    # Params & Optimization
    params:         Any
    opt_state:      Any

    # Preprocessor
    prep_state:     Any

    # Metrics
    metric_eprew:  jax.Array
    metric_return: MetricAggregator
    metric_update: MetricAggregator


def collect_trajectories(init_agent: AgentState, conf: AgentConfig, length: int, is_eval: bool = False) -> Tuple[AgentState, OnPolicyTrajectories]:

    def _collect_step(_agent: AgentState, _input=None) -> Tuple[AgentState, OnPolicyTrajectories]:
        new_key, policy_key = jax.random.split(_agent.key)

        # Preprocess obs
        new_prep_state, obs = conf.algo_cls.preprocess_obs(_agent.prep_state, conf.env_cls.observe(_agent.env_states))

        # Select actions
        new_policy_carry, (act, policy_output) = conf.policy_cls.apply(_agent.params, _agent.policy_carry, obs, _agent.env_is_new_episode,
                                                                       key=policy_key, is_eval=is_eval,
                                                                       method=conf.policy_cls.infer)

        # Step environment
        new_env_states, rew, done = conf.env_cls.step(_agent.env_states, act)
        assert done.dtype == jnp.bool_, "Done flag must be boolean."

        # Update metrics (episodic return)
        new_metric_eprew  = _agent.metric_eprew + rew
        new_metric_return = MetricAggregator.push(_agent.metric_return, jnp.sum(jnp.where(done, new_metric_eprew, 0)), count=jnp.sum(done))
        new_metric_eprew  = jnp.where(done, 0, new_metric_eprew)

        # Preprocess reward
        new_prep_state, rew = conf.algo_cls.preprocess_rew(new_prep_state, rew)

        # reset done envs
        # Reference: brax / envs / wrapper.py
        def _where_done(x, y):
            done = new_env_states.done
            done = done.reshape([-1] + [1] * (x.ndim - 1))
            return jnp.where(done, x, y)

        new_env_states = jax.tree_map(_where_done, _agent.env_reset_pool, new_env_states)

        return _agent.replace(
            key=new_key,
            # Env
            env_is_new_episode=done,
            env_states=new_env_states,
            # Policy
            policy_carry=new_policy_carry,
            # Preprocess
            prep_state=new_prep_state,
            # Metrics
            metric_eprew=new_metric_eprew,
            metric_return=new_metric_return,
        ), OnPolicyTrajectories(
            # Env
            obs=obs, rew=rew, done=done,
            is_new_episode=_agent.env_is_new_episode,
            # Policy
            carry=_agent.policy_carry,
            policy_output=policy_output
        )

    return jax.lax.scan(_collect_step, init_agent, None, length=length)


def update_policy(agent: AgentState, trajectories: OnPolicyTrajectories, conf: AgentConfig) -> AgentState:
    key, new_key = jax.random.split(agent.key)

    # Loss function
    def _loss_fn(params: Any):
        _, (_, policy_output) = conf.policy_cls.apply(params, trajectories.carry, trajectories.obs, trajectories.is_new_episode)

        return conf.algo_cls.loss_fn(key, policy_output, trajectories)

    # Update policy
    grads, metrics            = jax.grad(_loss_fn, has_aux=True)(agent.params)
    new_params, new_opt_state = conf.algo_cls.gradient_update(agent.params, agent.opt_state, grads)

    # Aggregate metrics
    new_metric_update = agent.metric_update
    if new_metric_update is None:
        new_metric_update = MetricAggregator.init(metrics)
    new_metric_update = MetricAggregator.push(new_metric_update, metrics)

    return agent.replace(
        key=new_key,
        # Policy
        params=new_params,
        # Optimizer
        opt_state=new_opt_state,
        # Metrics
        metric_update=new_metric_update
    )


@partial(jax.jit, static_argnames=["conf"])
def train_init(seed: int, conf: AgentConfig) -> AgentState:
    key_env, key_policy, key_train = jax.random.split(jax.random.PRNGKey(seed), 3)

    # init env
    env_reset_pool     = conf.env_cls.reset(jax.random.split(key_env, conf.train.batch_size))
    env_is_new_episode = jnp.ones(conf.train.batch_size, jnp.bool_)
    # get sample obs
    sample_obs         = conf.env_cls.observe(env_reset_pool)
    sample_rew         = env_reset_pool.reward

    # init policy carry + params
    policy_carry  = conf.policy_cls.initial_carry(conf.train.batch_size)
    policy_params = conf.policy_cls.init(
        {"params": key_policy},
        policy_carry, sample_obs, env_is_new_episode,
        method=conf.policy_cls.infer
    )

    # init opt
    opt_state = conf.algo_cls.gradient_init(policy_params)

    return AgentState(
        key=key_train,
        # Env
        env_states=env_reset_pool,
        env_reset_pool=env_reset_pool,
        env_is_new_episode=env_is_new_episode,
        # Policy
        policy_carry=policy_carry,
        params=policy_params,
        opt_state=opt_state,
        # Preprocessor
        prep_state=conf.algo_cls.preprocess_init(sample_obs, sample_rew),
        # Metrics
        metric_eprew=jnp.zeros_like(sample_rew),
        metric_return=MetricAggregator.init(jnp.zeros((), sample_rew.dtype)),
        metric_update=None
    )


# TODO: Multi GPU support
@partial(jax.jit, donate_argnums=(0,), static_argnames=["conf"])
def train_step(agent: AgentState, conf: AgentConfig) -> AgentState:
    agent, trajectories = collect_trajectories(agent, conf, conf.train.collect_len)
    agent               = update_policy(agent, trajectories, conf)

    return agent


@partial(jax.jit, donate_argnums=(0,))
def train_pop_metrics(agent: AgentState) -> Tuple[dict, AgentState]:
    new_metric_return, popped_return = MetricAggregator.pop(agent.metric_return)
    new_metric_update, popped_update = MetricAggregator.pop(agent.metric_update)

    return agent.replace(
        metric_return=new_metric_return,
        metric_update=new_metric_update
    ), {"return": popped_return, **popped_update}


# @partial(jax.jit, static_argnames=["conf"])
def eval_step(agent: AgentState, conf: AgentConfig) -> dict:
    # Create a virtual agent for evaluation
    # FIXME: How to do with policy carry ?
    eval_agent = agent.replace(
        # Reset env (All new episode)
        env_states=agent.env_reset_pool,
        env_is_new_episode=jnp.ones(conf.train.batch_size, jnp.bool_),
        # Reset metrics
        metric_eprew=jnp.zeros_like(agent.metric_eprew),
        metric_return=MetricAggregator.init(agent.metric_return.sum)
    )
    print("eval_agent.metric_return.sum:", eval_agent.metric_return.sum)
    # Eval return by collecting trajectories
    eval_agent, _    = collect_trajectories(eval_agent, conf, conf.env_cls.max_episode_length, is_eval=True)
    _, popped_return = MetricAggregator.pop(eval_agent.metric_return)
    print("type pop_return:",type(popped_return),"type metric_return:", type(eval_agent.metric_return))
    print("count:",jax.device_get(eval_agent.metric_return.count))
    print( "sum:", jax.device_get(eval_agent.metric_return.sum))
    return {"eval_return": popped_return}


def main(conf):
    conf = OmegaConf.merge({
        # Seed
        "seed": 0,

        # Task
        "env": "BraxVector",
        "env_conf": {
            "name": "humanoid"
        },

        # Policy
        "policy": "LSTM",
        "policy_conf": {},

        # Algorithm
        "algo": "PPO",
        "algo_conf": {},

        # Training
        "train_conf": {}
    }, conf)

    # Naming
    conf = OmegaConf.merge({
        "project_name": f"RecPG {conf.env}",
        "run_name":     f"{conf.algo} {conf.policy} {conf.seed} {time.strftime('%H:%M %m-%d')}"
    }, conf)

    # Create agent config
    env_cls    = ENVIRONMENTS[conf.env](**conf.env_conf)
    policy_cls = POLICIES[conf.policy](act_dims=env_cls.action_size, **conf.policy_conf)
    algo_cls   = ALGORITHMS[conf.algo](**conf.algo_conf)

    agent_conf = AgentConfig(
        env_cls=env_cls,
        policy_cls=policy_cls,
        algo_cls=algo_cls,

        train=TrainConfig(**conf.train_conf)
    )

    # Logging
    if "log_group" in conf:
        wandb.init(reinit=True, project=f"(G) {conf.project_name}", group=conf.log_group, name=str(conf.seed), config=OmegaConf.to_container(conf))
    else:
        wandb.init(reinit=True, project=conf.project_name, name=conf.run_name, config=OmegaConf.to_container(conf))

    # Training loop
    agent_state = train_init(conf.seed, agent_conf)
    for num_iter in tqdm(range(1, agent_conf.train.max_iteration + 1)):
        agent_state = train_step(agent_state, agent_conf)

        # Log train metrics
        if 0 == (num_iter % agent_conf.train.metric_log_every):
            agent_state, metrics = train_pop_metrics(agent_state)
            wandb.log(jax.device_get(metrics), step=num_iter)
        
        # Evaluate
        if 0 == (num_iter % agent_conf.train.metric_eval_every):
            eval_metrics = eval_step(agent_state, agent_conf)
            wandb.log(jax.device_get(eval_metrics), step=num_iter)


if __name__ == "__main__":
    main(OmegaConf.from_cli())
