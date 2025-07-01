import jax
import jax.numpy as jnp
import numpy as np

import flax
from flax.core import FrozenDict

import optax

from typing import Any, Tuple, Union, List
from functools import partial

from . import core
from .evo_config import EvoConfig, partition_optim_cls


@flax.struct.dataclass
class EvoState:
    """Hold network information during evolution training"""

    key: jax.Array
    
    # For Network optimization
    # All network parameters are Real numbers
    params: Any
    opt_state: Any
    fitness_past: float
    annealing: float


@flax.struct.dataclass
class EvoStateInitConfig:
    """Configure Starting point of training"""

    input_shape: tuple[int, ...]
    from_scratch: bool = True

    # Hyper parameters for converting pretrained params
    scale_sigma: float = 5.0  # How many sigmas puts into [0, 1] interval


def state_init(
        key: jax.Array,
        evo_conf: EvoConfig,
        init_conf: EvoStateInitConfig,
        pre_train_params: Any = None,
) -> EvoState:
    """
    Initialize a Evo Training state.
    Each device have a same copy of parameter, but have different rng keys.
    Already jit-ed. No need to jit again.
    """
    
    local_device_count = jax.local_device_count()
    key = jax.random.split(key, local_device_count)
    return jax.pmap(lambda k: _param_init(k, evo_conf, init_conf, pre_train_params))(key)


@partial(jax.jit, static_argnums=(1, 2,))
def _param_init(key, evo_conf: EvoConfig, init_conf: EvoStateInitConfig, pre_train_params=None, init_fitness=0.1) -> EvoState:
    if init_conf.from_scratch:
        # From Scratch
        network_cls = evo_conf.network_cls
        network_params = network_cls.init(key, jnp.zeros(init_conf.input_shape))  # key is not really used. init should be determinstic.
        params = FrozenDict({
            "params": core.bool_theta_to_rho(network_params["params"], p_dtype=evo_conf.p_dtype),
        })

    else:
        # From pretrained parameters
        if not isinstance(pre_train_params, dict):
            raise Exception("Require a pretrained parameters as a dict to initialize")

        def _body(keypath, x):
            if isinstance(x, jax.Array):
                return x

            # Assume bias = 0 and mean = 0
            pre_kernel = x["kernel"]
            standard_dev = jnp.std(pre_kernel)
            target_scale = 0.5 / init_conf.scale_sigma
            scale = standard_dev / target_scale
            kernel = 0.5 + pre_kernel / scale
            kernel = jnp.clip(kernel, 0.001, 0.999)
            kernel = jnp.matrix_transpose(kernel)  # For left & right matmul difference between torch and jax
            return {
                core.CONN_KERNEL: kernel,
                "scale": scale,
            }

        def _is_leaf(x):
            if isinstance(x, jax.Array):
                return True
            if "kernel" in x.keys():
                return True
            return False

        params = jax.tree_util.tree_map_with_path(_body, pre_train_params, is_leaf=_is_leaf)
        params = FrozenDict({"params": params})

    optim_cls = partition_optim_cls(evo_conf, params)

    opt_state = optim_cls.init(params)

    return EvoState(
        key=key,
        params=params,
        opt_state=opt_state,
        fitness_past=init_fitness,
        annealing=1,
    )


def state_update(
        evo_state: EvoState,
        evo_conf: EvoConfig,
        data_loader: Any,
) -> Tuple[EvoState, jax.Array]:
    """
    Main training function. Update params ONE step with given data batch.
    Return a new evolution state, and the training fitness.
    Already jit-ed. No need to jit again.
    """

    # Gradient Accuumulation, run smaller subpops
    key = evo_state.key
    annealing = evo_state.annealing
    params = evo_state.params
    opt_state = evo_state.opt_state
    grads, tot_fitness = _init_grads(params)
    fitness_past = evo_state.fitness_past
    data_batch = data_loader.get_batch()
    for i in range(evo_conf.num_subpop):
        key, grads, tot_fitness = _fwd_and_grads(
            key,
            params,
            grads,
            data_batch,
            evo_conf,
            tot_fitness,
            fitness_past)
    tot_fitness /= evo_conf.epoch_pop_size
    fitness_past = evo_conf.e*fitness_past+(1-evo_conf.e)*np.sum(tot_fitness)
    # Perform gradient step
    # Grads is only meaned by subpop_size. Therefore need to divide device_cnt * num_subpop
    params, opt_state = _grad_step(grads, params, opt_state, evo_conf, annealing)
    
    if evo_conf.reset>0:
        params = core.reset(key, params, evo_conf.reset*annealing)
    

    return EvoState(
        key=key,
        params=params,
        opt_state=opt_state,
        fitness_past = fitness_past,
        annealing = annealing * jnp.exp(evo_conf.annealing_rate),
    ), tot_fitness


@jax.pmap
def _init_grads(params: FrozenDict):
    grads = jax.tree_map(jnp.zeros_like, params)
    tot_fitness = jnp.zeros(())
    return grads, tot_fitness

@partial(jax.pmap, in_axes=(0, 0, 0, 0, None, 0,0), static_broadcasted_argnums=(4,), axis_name='i')
def _fwd_and_grads(
        key: jax.Array,  # 0
        params: FrozenDict,  # 1
        grads: FrozenDict,  # 2
        data_batch: Union[FrozenDict, jax.Array],  # 3
        evo_conf: EvoConfig,  # 4, static & broadcasted
        tot_fitness: jax.Array,# 5
        fitness_past: float):  #6

    # Prepare the network batch
    batch_size = (evo_conf.subpop_size,)
    next_key, key = jax.random.split(key)
    theta = core.sample_bernoulli_param(key, params, batch_size)


    # Compute the forward pass
    input_batch, label_batch = data_batch
    fwd_axes = core.evo_tree_axes(theta)

    logits = jax.vmap(evo_conf.network_cls.apply, in_axes=(fwd_axes, 0))(theta, input_batch)
    # jax.debug.print("neurons output:{} {}",jnp.mean(logits),jnp.std(logits))
    # fitness_acc = evo_conf.acc_cls(logits, label_batch)
    fitness = evo_conf.fitness_cls(logits, label_batch)
    
    tot_fitness += jnp.sum(fitness)
    

    # Gather key and fitness information from other devices
    key = jax.lax.all_gather(key, axis_name='i')
    fitness = jax.lax.all_gather(fitness, axis_name='i')

    #TODO
    if evo_conf.fitness_transform == "crt":
        fitness = core.centered_rank_transform(fitness)
    elif evo_conf.fitness_transform == "history":
        fitness -= fitness_past
    elif evo_conf.fitness_transform == "sigmoid":
        fitness = core.sigmoid((fitness - fitness_past)*10)
        
    

    for device_idx in range(jax.local_device_count()):
        _key = key[device_idx]
        _fitness = fitness[device_idx]

        _theta = core.sample_bernoulli_param(_key, params, batch_size)
        _grads = core.nes_grad(_fitness, _theta, params)
        grads = jax.tree_map(jnp.add, grads, _grads)

    return next_key, grads, tot_fitness


@partial(jax.pmap, in_axes=(0, 0, 0, None, 0), static_broadcasted_argnums=(3,))
def _grad_step(grads: FrozenDict, params: FrozenDict, opt_state: optax.OptState, evo_conf: EvoConfig, annealing: float):
    opt_cls = evo_conf.optim_cls
    eps = evo_conf.eps

    # Grads is only meaned by subpop_size. Therefore need to divide device_cnt * num_subpop
    grads = jax.tree_map(lambda x: x / (evo_conf.num_subpop * jax.device_count()), grads)

    # If trained by loss, minimize the loss.
    if not evo_conf.maximize_fitness:
        grads = jax.tree_map(lambda x: -x, grads)

    # Gradient Step
    updates, new_opt_state = opt_cls.update(grads, opt_state, params)

    if evo_conf.weight_transform == "weight_decay":
        updates = core.weight_decay(updates, params, evo_conf.weight_decay_rate, annealing)

    new_params = optax.apply_updates(params, updates)

    # Clip to Bernoulli range with exploration
    new_params = jax.tree_map(lambda p: jnp.clip(p, eps, 1 - eps), new_params)

    return new_params, new_opt_state


@partial(jax.jit, static_argnums=1)
def evaluation(
        evo_state: EvoState,
        evo_conf: EvoConfig,
        data_batch: Any,
):
    """
    Main testing function. 
    Return the testing fitness.
    """
    params = evo_state.params
    theta = core.determinstic_param(params)
    input_batch, label_batch = data_batch
    logits = jax.vmap(evo_conf.network_cls.apply)(theta, input_batch)
    fitness = jnp.mean(evo_conf.acc_cls(logits, label_batch))
    return fitness

def test(
        evo_state: EvoState,
        evo_conf: EvoConfig,
        data_batch: Any,
):
    """
    evaluation in torch
    """
    params = evo_state.params
    theta = core.determinstic_param(params)

    input_batch, label_batch = data_batch
    input_batch = np.expand_dims(input_batch, axis=0).repeat(jax.device_count(), axis=0)
    label_batch = np.expand_dims(label_batch, axis=0)
    logits = jax.vmap(evo_conf.network_cls.apply)(theta, input_batch)
    fitness = evo_conf.acc_cls(logits[0], label_batch[0])

    return fitness