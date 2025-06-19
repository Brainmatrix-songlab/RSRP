# Evolving Connectivity for Recurrent Spiking Neural Networks Repository

This repository contains the implementation of the paper [Evolving Connectivity for Recurrent Spiking Neural Networks](https://arxiv.org/abs/2305.17650). It includes the Evolutionary Connectivity (EC) algorithm, Recurrent Spiking Neural Networks (RSNN), and the Evolution Strategies (ES) baseline implemented in JAX.

## Getting Started

### Prerequisites

1. [Install JAX](https://github.com/google/jax#installation)

2. [Install W&B](https://github.com/wandb/wandb) and log in to your account to view metrics

3. Install the required dependencies:

```bash
conda create --name <env> --file requirements.txt
```

### Precautions

- Brax v1 is required (`brax<0.9`) to reproduce our experiments. Brax v2 has completely rewritten the physics engine and adopted a different reward function.
- Due to the inherent numerical stochasticity in Brax's physics simulations, variations in results can occur even when using a fixed seed.

## Usage

### Training EC with RSNN

To set parameters, use the command-line format of [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#id15). For example:

```
python ec_sweep_v2.py
#or you want to sweep
python ec_sweep_v2.py sweep
```

### Training EC with RSNN-Reservior

To set parameters, use the command-line format of [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#id15). For example:

```
python ec_frozen_recurrent.py
#or you want to sweep
python ec_frozen_recurrent.py sweep
```

