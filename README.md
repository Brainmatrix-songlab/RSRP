# Reward-Optimizing Learning using Stochastic Release Plasticity

This repository provides the implementation of RSRP, a biologically inspired synaptic plasticity rule that models synaptic transmission as a parameterized Bernoulli distribution and optimizes release probabilities using natural gradient estimation. This rule enables robust and effective learning in both reinforcement learning and supervised learning settings, with plausible neurobiological mechanisms such as reward-modulated synaptic updates and EI-balanced networks.

## 🌟 Highlight
+ ✔️ A unified plasticity rule for both RL and classification tasks
+ ✔️ Competitive with PPO in reinforcement learning
+ ✔️ Near backpropagation-level accuracy in classification
+ ✔️ Support for biologically plausible networks (e.g., EI balance, sparse reward)
+ ✔️ Implemented entirely in JAX, with support for OpenAI Gym and Brax

## 📁 Repository Structure
```
.
├── Cartpole-main/
│   ├── EvolvingConnectivity-master/   # CartPole tasks (RSRP, ES, PPO-LSTM)
│   └── Netpyne-STDP-master/           # CartPole task using R-STDP
├── RSRP_singleGPU/                    # Humanoid tasks with RSRP and ES
├── RSRP_multiGPU/                     # Classification tasks (MNIST, CIFAR)
├── PPO-brax/                          # PPO-LSTM implementation using Brax
├── DiehlCook_MNIST_evaluation/        # Baseline STDP model on MNIST              
└── README.md
```

## 📦 Environment Setup
1. Install JAX
Follow instructions at: https://github.com/google/jax#installation
2. Install W&B for logging
```
pip install wandb
wandb login
```
3. Install other dependencies
```
pip install -r requirements.txt
```

## 🚀 Usage Instructions
### 🧠 Reinforcement Learning Tasks
#### Cartpole task
+ RSRP: Figure 2A (1) in Folder Cartpole-main/EvolvingConnectivity-master

Single run
```
python ec_cartpole.py
```
Multiple runs
```
python ec_cartpole.py sweep
```

+ RSRP-Reservior: Figure 2A (2) in Folder Cartpole-main/EvolvingConnectivity-master      

Single run
```
python ec_resevior.py
```
Multiple runs
```
python ec_resevior.py sweep
```

+ ES: Figure 2A (3) in Folder Cartpole-main/EvolvingConnectivity-master   

Single run
```
python es_cartpole.py
```
Multiple runs
```
python es_cartpole.py sweep
```

+ PPO-LSTM: Figure 2A (4) in Folder Cartpole-main/EvolvingConnectivity-master   

Single run
```
python ppo-gym-cartpole.py
```
Multiple runs
```
python ppo-gym-cartpole.py sweep
```

+ R-STDP: Figure 2A (5) in Folder Cartpole-main/Netpyne-STDP-master          
```
nrnivmodl mod
python neurosim/sim.py run
```

#### Humanoid task
+ RSRP: Figure 2B (1) and Figure 3 blue curve in Folder RSRP_singleGPU

Single run
```
python RSRP_sweep.py
```
Multiple runs
```
python RSRP_sweep.py sweep
```

+ RSRP-Reservior: Figure 2B (2) in Folder RSRP_singleGPU

Single run
```
python RSRP_frozen_recurrent.py
```
Multiple runs
```
python RSRP_sweep.py sweep
```

+ ES: Figure 2B (3) in Folder RSRP_singleGPU

Single run
```
python ES_sweep.py
```
Multiple runs
```
python ES_sweep.py sweep
```

+ PPO-LSTM: Figure 2B (4) and Figure 3 red curve in Folder PPO-brax
```
python exp_launcher.py include=experiment_conf/lstm.yaml
```

### 📊 Supervised Learning Tasks
#### MNIST task
+ STDP: Figure 6A(STDP curve) in Folder DiehlCook_MNIST_evaluation
```
python DiehlCook_spiking_MNIST.py
```
+ RSRP: Figure 6 and Figure 7 in Folder RSRP_multiGPU
```
python ec_mnist.py
```
#### CIFAR task 
+ RSRP: Figure 6 and Figure 7 in Folder RSRP_multiGPU
```
python ec_cifar.py
```

## 📅 Citation
If you find this codebase or our method useful in your research, please consider citing:
```

```

## 📩 Contact
If you have any questions or are interested in collaborating, please feel free to contact us directly via email:
```
syh18@mails.tsinghua.edu.cn
songsen@tsinghua.edu.cn
```
