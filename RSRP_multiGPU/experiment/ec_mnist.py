XLA_PYTHON_CLIENT_MEM_FRACTION=.90

import sys
import os

import ec
from ec import EvoState, EvoConfig, EvoStateInitConfig, NumpyDataLoader, PreAllocatedDataLoader, modules, optim, core, conf_trans, utils

import time

import jax
import jax.numpy as jnp
import numpy as np

from torchvision import datasets, transforms

from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
import optuna

sys.path.append('/home/liaowantong/EC')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ec.utils.init_env()

conf = OmegaConf.from_cli()
conf = OmegaConf.merge(conf, {
    # Task
    "seed": 1,
    "task": "renew",
    "data": "MNIST",
    "network_type": "MLP",
    "wandb": False,

    # Network
    "network_size": 64,

    # Train
    "total_generations": 5000,
    "pop_size": 20480,
    "subpop_num": 2,                    # 内存不足时串行
    "device_num": jax.device_count(),   # 多卡并行
    "reward": "softrecall",
    "batch_size": 64,
    "optim": "sgd",
    "momentum": 0.9,
    "lr": 4,
    "fitness_transform": "crt",

    "e": 0,
    "layer_lr": "none",
    "weight_transform": "none",
    "weight_decay_rate": 0,
    "annealing_rate": 0,
    "reset": 0,
    "EI_balance":"1:1_input",
})

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(0.1307,0.3081),
                                transforms.Lambda(lambda x: x.view(-1))
                                ])
train_dataset = datasets.MNIST('~/data/mnist/', download=False, train=True, transform=transform)
test_dataset = datasets.MNIST('~/data/mnist/', download=False, train=False, transform=transform)

evo_conf_test = EvoConfig(batch_size=10000//jax.device_count(), epoch_pop_size=jax.device_count(), subpop_size=1)
test_data_loader = PreAllocatedDataLoader(dataset=test_dataset, evo_conf=evo_conf_test, train=False)

def main(conf):
    conf = OmegaConf.merge(conf, {
    "project_name": f"EC-{conf.task}",
    "run_name": f"lr={conf.lr},ps={conf.pop_size},{conf.seed}",
    "ps_bs": conf.pop_size*conf.batch_size,
    })

    evo_conf = EvoConfig(
        network_cls=ec.modules.MLP((conf.network_size, 10,)),
    )
    evo_conf = conf_trans(evo_conf,conf)
    utils.setup_seed(conf.seed)
    train_data_loader = PreAllocatedDataLoader(dataset=train_dataset, evo_conf=evo_conf, train=True)

    # Initialization
    init_conf = EvoStateInitConfig(
        input_shape=(2, *train_data_loader.data_shape[0]),
    )
    key = jax.random.PRNGKey(conf.seed)

    evo_state = ec.state_init(key, evo_conf, init_conf)

    evo_conf = evo_conf.replace(
        optim_cls=ec.partition_optim_cls(evo_conf, evo_state.params),
    )
    if conf.wandb:
        wandb.init(reinit=True, project=conf.project_name,
                   name=conf.run_name,
                   config=OmegaConf.to_container(conf))
        
    # Main Training Procedure
    for step in tqdm(range(1, conf.total_generations + 1)): 
        evo_state, fitness = ec.state_update(evo_state, evo_conf, train_data_loader)
        if (step-1)% 20 == 0:
            eval = ec.evaluation(evo_state, evo_conf, test_data_loader.get_batch())
        # print(jnp.mean(evo_state.params["params"]["Dense_0"]["ConnKernel"]),jnp.mean(evo_state.params["params"]["Dense_1"]["ConnKernel"]))
        # print(jnp.mean(evo_state.params["params"]["Dense_input_0"]["ConnKernel"]),jnp.mean(evo_state.params["params"]["Dense_EI_0"]["ConnKernel"]))
        metrics = {
            "fitness": np.sum(fitness),
            # "Entrophy": ec.metrics.avg_entrophy(evo_state.params),
            "evaluation": eval,
        }
        if conf.wandb:
            wandb.log(metrics, step=step)
        
    for data_batch in test_data_loader:
        metrics["test_acc"] = ec.test(evo_state, evo_conf, data_batch)
    print("test=",metrics["test_acc"])



    if conf.wandb:
        wandb.log(metrics, step=step)
        wandb.finish()

    return metrics


def sweep(seed: int, conf_override: OmegaConf):
    def _objective(trial: optuna.Trial):
        conf = OmegaConf.merge(conf_override, {
            # "seed": trial.suggest_categorical("seed", [0,1,2,3,4,5,6,7,8,9,10]),
            # "EI_balance":trial.suggest_categorical("EI_balance", ["1:1","1:1_input"]),
            "reward":       trial.suggest_categorical("reward",["softrecall","accuracy","cross_entropy","recall"]),
            # "fitness_transform":    trial.suggest_categorical("fitness_transform", ["crt","none","history"]),
            # "optim":        trial.suggest_categorical("optim", ["sgd","momentum","adam"]),
            # "layer_lr": trial.suggest_categorical("layer_lr", ["entropy", "norm"]),
            # "lr":           trial.suggest_categorical("lr", [0.005,0.002,0.01,0.02,0.05,0.1,0.2,3,5,8]),
            # "weight_decay_rate": trial.suggest_categorical("weight_decay_rate", [0.0002, 0.001, 0.0005]),
            # "batch_size":   trial.suggest_categorical("batch_size", [1,2,4,8,16,32,64,128,256,512]),
            # "pop_size":     trial.suggest_categorical("pop_size", [512, 1024, 2048, 4196, 8392]),
            # "momentum":     trial.suggest_categorical("momentum",  [0.9, 0.95, 0.99]),
            # "e":            trial.suggest_categorical("e",[0.9,0.5,0.0])
        })
        metrics = main(conf)
        result = metrics["test_acc"]
        return result

    search_space = {"reward":["softrecall","accuracy","cross_entropy","recall"]}
    optuna.create_study(direction="maximize", sampler=optuna.samplers.GridSampler(search_space)).optimize(_objective)


if __name__ == "__main__":
    if hasattr(conf, "sweep"):
        sweep(conf.sweep, conf)
    else:
        main(conf)
