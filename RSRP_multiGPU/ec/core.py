from typing import Tuple
from functools import partial
import math

import jax
import jax.numpy as jnp

import flax
from flax.core import FrozenDict

FWD_IN_AXES = (FrozenDict({
    "params": 0,  # bool params
}), 0)  # input batch

CONN_KERNEL = "ConnKernel"  # Used to mark that a kernel is an evolutionable kernel

"""
DNN networks have a variable FrozenDict like 
{"params": .. , "others": ..}

EC networks have a variable FrozenDict like 
{"params": 
    {"layer1":
        {"ConnKernel": BoolArray, "others": FPArray},
    "Other layer": .. },
"others": ..}

Only ConnKernel is evolutionable.
"""

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def _trainable(keypath) -> bool:
    if any(CONN_KERNEL in k.key for k in keypath):
        return True
    return False


def evo_tree_axes(params: FrozenDict):
    def _body(keypath, x):
        if _trainable(keypath):
            return 0
        else:
            return None

    res = jax.tree_util.tree_map_with_path(_body, params)
    return res


def bool_theta_to_rho(params: FrozenDict, init_prob: float = 0.5, p_dtype: jnp.dtype = jnp.bfloat16) -> FrozenDict:
    """
    Convert the bool networks connections into fixed initial probs, with same shape. 
    ONLY ConnKernel is converted.
    """

    def _body(keypath, x):
        if _trainable(keypath):
            return jnp.full_like(x, init_prob, p_dtype)
        else:
            return x

    return jax.tree_util.tree_map_with_path(_body, params)


def sample_bernoulli_param(key: jax.Array, params: FrozenDict, batch: Tuple[int, ...]) -> FrozenDict:
    """
    Sample parameters from Bernoulli distribution. Only trainable parameters is sampled.
    TODO: Better efficiency sampling method. Implementing with jax.random.bernoulli now. 
    TODO: Warning: the sampling process could be a memory bottleneck. Results are lower mem dtype, but intermediates could be higher mem dtype.  
    """

    num_vars = len(jax.tree_util.tree_leaves(params))
    treedef = jax.tree_util.tree_structure(params)
    
    all_keys = jax.random.split(key, num=num_vars)

    def _body(keypath, p, k):
        if _trainable(keypath):
            return jax.random.bernoulli(k, p, (*batch, *p.shape))
        else:
            return p

    noise = jax.tree_util.tree_map_with_path(
        _body,
        params,
        jax.tree_util.tree_unflatten(treedef, all_keys)
    )

    return noise


def determinstic_param(params: FrozenDict) -> FrozenDict:
    """Deterministic evaluation, using p > 0.5 as True, p <= 0.5 as False"""

    def _body(keypath, p):
        if _trainable(keypath):
            return p > 0.5
        else:
            return p

    return jax.tree_util.tree_map_with_path(_body, params)


def centered_rank_transform(x: jax.Array) -> jax.Array:
    """
    Centered rank transfrom: https://arxiv.org/pdf/1703.03864.pdf
    Scaled to zero mean and variance = 1.
    """

    scale = math.sqrt(12.0)

    shape = x.shape
    x = x.ravel()

    x = jnp.argsort(jnp.argsort(x))
    x = x / (len(x) - 1) - 0.5
    x = x * scale
    return x.reshape(shape)


def p_centered_rank_transform(x: jax.Array, axis_name, axis=1) -> jax.Array:
    """
    Centered rank transfrom: https://arxiv.org/pdf/1703.03864.pdf.
    x is distributed across devices, axis_name is device axis.
    Result is replicated to all devices.
    """

    x = jax.lax.all_gather(x, axis_name=axis_name, axis=axis)
    x = centered_rank_transform(x)

    return x


def p_split(key: jax.Array, num: int = 2) -> Tuple[jax.Array, ...]:
    """
    Spilt for sharded key, into num new sharded keys.
    """

    return _p_split(key, num)


@partial(jax.pmap, in_axes=(0, None), static_broadcasted_argnums=(1,))
def _p_split(key: jax.Array, num: int) -> Tuple[jax.Array, ...]:
    res = []
    for i in range(num - 1):
        key, subkey = jax.random.split(key)
        res.append(subkey)
    res.append(key)

    return tuple(res)


def nes_grad(fitness: jax.Array, theta: FrozenDict, rho: FrozenDict) -> FrozenDict:
    """
    Compute EC grad estimation, grad = -R * (theta - rho)
    """

    def _nes_grad(keypath, _rho, _theta):
        if _trainable(keypath):
            R = fitness.reshape((-1,) + (1,) * (_theta.ndim - 1)).astype(_rho.dtype)
            return -jnp.mean(R * _theta, axis=0)   #-jnp.mean(R * (_theta - _rho), axis=0)
        else:
            return jnp.zeros_like(_theta)

    return jax.tree_util.tree_map_with_path(_nes_grad, rho, theta)


def weight_decay(grads: FrozenDict, params: FrozenDict, decay_rate, annealing) -> FrozenDict:
    """
    Compute EC grad estimation, grad = -R * (theta - rho)
    """

    def _weight_decay(keypath, _grads, _params):
        if _trainable(keypath):
            decay = annealing * decay_rate * (_params - 0.5)
            return _grads - decay
        else:
            return jnp.zeros_like(_grads)

    return jax.tree_util.tree_map_with_path(_weight_decay, grads, params)

def reset(key: jax.Array, params: FrozenDict, p: float) -> FrozenDict:
    """
    reset p persent of params to 0.5 
    """
    num_vars = len(jax.tree_util.tree_leaves(params))
    treedef = jax.tree_util.tree_structure(params)
    all_keys = jax.random.split(key[0], num=num_vars)

    def _reset(keypath,_params, k):
        mask = jax.random.bernoulli(k, p, _params.shape)
        return _params - mask*(_params-0.5)

    return jax.tree_util.tree_map_with_path(_reset, params,jax.tree_util.tree_unflatten(treedef, all_keys))