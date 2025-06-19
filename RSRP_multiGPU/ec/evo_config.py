import jax
import jax.numpy as jnp
import flax
import optax

from flax.core import FrozenDict

from typing import Any, Callable, Dict

from . import core


@flax.struct.dataclass
class EvoConfig:
    # Network, optim & other instance
    network_cls: Callable = None
    optim_cls: Any = None
    fitness_cls: Callable = None  # Fitness used as training return
    acc_cls: Callable = None
    fitness_transform: Any = None
    e: float = 1
    metric_cls: Dict = None  # Metric used for testing
    forward_cls: Callable = None  # Forward pass function of the network

    # Hyperparameters - Training
    optim_param: Any = None
    batch_size: int = 64  # Batch is mini-batch for one network
    epoch_pop_size: int = 1024
    eps: float = 1e-3  # Clip params into [eps, 1-eps]
    maximize_fitness: bool = True  # Maximize = False for loss

    weight_transform: Any = None
    weight_decay_rate: float = 0
    annealing_rate: float = 0
    reset: float = 0

    # Hyperparameters - Engineering
    # settings
    p_dtype: jnp.dtype = jnp.float32
    multi_device: bool = True
    subpop_size: int = 1024  # Bigger subpop_size means higher parallel

    # useful numbers, infered from settings
    @property
    def device_pop_size(self):
        return self.epoch_pop_size // jax.device_count()

    @property
    def num_subpop(self):
        return self.device_pop_size // self.subpop_size

    @property
    def pop_batch_size(self):
        return jax.local_device_count() * self.subpop_size * self.c

    @property
    def pop_batch_shape(self):
        return (jax.local_device_count(), self.subpop_size, self.batch_size)

    def sanity_check(self) -> bool:

        device_count = jax.device_count()
        if self.epoch_pop_size % device_count != 0:
            raise Exception(f"Epoch pop size = {self.epoch_pop_size}, Device count = {device_count}, Not divisible")

        if self.device_pop_size % self.subpop_size != 0:
            raise Exception(
                f"Device pop size = {self.device_pop_size}, Subpop size = {self.subpop_size}, Not divisible")

        return True


def partition_optim_cls(evo_conf: EvoConfig, params: FrozenDict):
    """
    Return a optimizers to ignore untrainable parameters.
    """

    partition_optim = {'trainable': evo_conf.optim_cls, "frozen": optax.set_to_zero()}
    param_partitions = FrozenDict(flax.traverse_util.path_aware_map(
        lambda path, v: 'trainable' if core.CONN_KERNEL in path else 'frozen',
        params))
    optim_cls = optax.multi_transform(partition_optim, param_partitions)

    return optim_cls


def batch_fwd(evo_conf: EvoConfig, theta: FrozenDict, data_batch: Any) -> jax.Array:
    """
    Compute the forward, with batched inputs, typically work for classification tasks.
    databatch: [input_batch, label_batch]
    """

    network_cls = evo_conf.network_cls
    fitness_cls = evo_conf.fitness_cls
    input_batch, label_batch = data_batch
    fwd_axes = core.evo_tree_axes(theta)

    logits = jax.vmap(network_cls.apply, in_axes=(fwd_axes, 0))(theta, input_batch)
    fitness = fitness_cls(logits, label_batch)
    return fitness


def pretrain_clm_batch_fwd(evo_conf: EvoConfig, theta: FrozenDict, data_batch: Any) -> jax.Array:
    """
    Compute the forward pass of clm tasks, using models inherit from transformers.FlaxPretrainModel
    """

    network_cls = evo_conf.network_cls
    fitness_cls = evo_conf.fitness_cls
    fwd_axes = core.evo_tree_axes(theta)

    def _fwd_fn(_theta, _input):
        return network_cls.__call__(params=_theta, **_input)

    outputs = jax.vmap(_fwd_fn, in_axes=(fwd_axes, 0))(theta, data_batch)
    fitness = fitness_cls(outputs.logits, data_batch['labels'], mean_axis=(-1, -2))
    return fitness

import ec
from ec import optim

def conf_trans(evo_conf,conf):

    evo_conf = evo_conf.replace(
        acc_cls=ec.metrics.accuracy,
        forward_cls=ec.batch_fwd,

        epoch_pop_size=conf.pop_size,
        batch_size=conf.batch_size,
        subpop_size=conf.pop_size//conf.device_num//conf.subpop_num,
    )

    # reward
    if conf.reward == "softrecall":
        evo_conf = evo_conf.replace(
            fitness_cls=ec.metrics.softrecall,
        )
    elif conf.reward == "softacc":
        evo_conf = evo_conf.replace(
            fitness_cls=ec.metrics.softacc,
        )
    elif conf.reward == "recall":
        evo_conf = evo_conf.replace(
            fitness_cls=ec.metrics.recall,
        )
    elif conf.reward == "accuracy":
        evo_conf = evo_conf.replace(
            fitness_cls=ec.metrics.accuracy,
        )
    elif conf.reward == "cross_entropy":
        evo_conf = evo_conf.replace(
            fitness_cls=ec.metrics.cross_entrophy,
        )
    # reward transform
    evo_conf = evo_conf.replace(
        fitness_transform=conf.fitness_transform,
        e=conf.e,
    )    
    # optimizer
    if conf.optim == "sgd":
        evo_conf = evo_conf.replace(
            optim_cls=optax.sgd(learning_rate=conf.lr),
        )
    elif conf.optim == "momentum":
        evo_conf = evo_conf.replace(
            optim_cls=optax.sgd(learning_rate=conf.lr, momentum=conf.momentum),
        )
    elif conf.optim == "adam":
        evo_conf = evo_conf.replace(
            optim_cls=optax.adam(learning_rate=conf.lr),
        )

    # layer lr
    if conf.layer_lr == "entropy":
        evo_conf = evo_conf.replace(
            optim_cls=optax.chain(optim.layer_lr_by_entropy(), evo_conf.optim_cls),
        )
    elif conf.layer_lr == "norm":
        evo_conf = evo_conf.replace(
            optim_cls=optax.chain(optim.layer_lr_by_norm(), evo_conf.optim_cls),
        )
    elif conf.layer_lr == "sqrtnorm":
        evo_conf = evo_conf.replace(
            optim_cls=optax.chain(optim.layer_lr_by_sqrtnorm(), evo_conf.optim_cls),
        )
    # wd & reset
    if conf.weight_transform == "weight_decay":
        evo_conf = evo_conf.replace(
            weight_transform=conf.weight_transform,
            weight_decay_rate=conf.weight_decay_rate,
            annealing_rate=conf.annealing_rate,
        )
    elif conf.weight_transform == "reset":
        evo_conf = evo_conf.replace(
            reset=conf.reset,
            annealing_rate=conf.annealing_rate,
        )
    # EI network
    if conf.EI_balance == "1:1":
        evo_conf = evo_conf.replace(
            network_cls=ec.modules.MLP_EI((conf.network_size, 10,),p=0.5),
        )
    elif conf.EI_balance == "1:1_input":
        evo_conf = evo_conf.replace(
            network_cls=ec.modules.MLP_EI_input((conf.network_size, 10,),p=0.5),
        )
    elif conf.EI_balance == "3:1":
        evo_conf = evo_conf.replace(
            network_cls=ec.modules.MLP_EI((conf.network_size, 10,),p=0.75),
        )
    elif conf.EI_balance == "2:1":
        evo_conf = evo_conf.replace(
            network_cls=ec.modules.MLP_EI((conf.network_size, 10,),p=2/3),
        )
    elif conf.EI_balance == "01":
        evo_conf = evo_conf.replace(
            network_cls=ec.modules.MLP_01((conf.network_size, 10,)),
        )
    
    return evo_conf
