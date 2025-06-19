from functools import partial
from typing import Any, Dict, Tuple, Union
import math
import os
import pickle

import torch
import random 
import numpy as np

import jax
import jax.numpy as jnp

import flax
from flax.core import FrozenDict

import logging

import builtins


def init_env():
    """
    Useful environment settings, for RNGs and BF16.
    Should be called early, just after imports.
    """

    # Use Threefry generator to makesure of identical across devices and shardings.
    # https://jax.readthedocs.io/en/latest/jax.random.html
    # jax.config.update("jax_default_prng_impl", "threefry")
    jax.config.update("jax_threefry_partitionable", 1)

    # Hack for resolving bfloat16 pickling issue https://github.com/google/jax/issues/8505
    builtins.bfloat16 = jnp.dtype("bfloat16").type


def init_logging(filename, file_lvl = logging.INFO, console_lvl = logging.WARNING):

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create a console handler that logs warning and above messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_lvl)  # Set the minimum log level for the console handler
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logging.basicConfig(level=file_lvl,
                    filename=filename,
                    filemode='w',
                    format='%(asctime)s - %(name)s - %(levelname)s\n%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    
    logging.getLogger().addHandler(console_handler)
    logging.warning(f"Logging info into {filename}")


def show_param_statistic(params: Union[FrozenDict, dict]) -> str:

    def _iter(p):
        yield from jax.tree_util.tree_leaves_with_path(p)

    iterator = _iter(params)

    res = ''

    for p, i in iterator:
        res += '{}   -full path = {}\n'.format(p[-1].key, [i.key for i in p])
        res += (f'    Shape: {i.shape}\n'
            f'    Mean: {jnp.mean(i):.8f}\n'
            f'    Max: {jnp.max(i):.8f}\n'
            f'    Min: {jnp.min(i):.8f}\n'
            f'    Variance: {jnp.var(i):.8f}\n'
            f'    STD: {jnp.sqrt(jnp.var(i)):.8f}\n')
        
    return res


def show_param_memory(params: Union[FrozenDict, dict]) -> str:

    def _calculate_array_size_in_bytes(array):
        """Calculate the size in bytes for a JAX array."""
        return array.size * array.dtype.itemsize

    sizes = jax.tree_map(_calculate_array_size_in_bytes, params)
    total_size_in_bytes = sum(jax.tree_leaves(sizes))
    gigabytes = total_size_in_bytes / (1024**3)
    megabytes = (gigabytes - int(gigabytes)) * 1024
    return f"{int(gigabytes)} G {int(megabytes)} M"

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True