#PPO-LSTM
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time
import numpy as np
import wandb
from omegaconf import OmegaConf
import optuna
import time
from tqdm import tqdm
import os
os.environ['WANDB_SILENT'] = 'true'
#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 2
T_horizon     = 20

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(4,64)
        self.lstm  = nn.LSTM(64,32)
        self.fc_pi = nn.Linear(32,2)
        self.fc_v  = nn.Linear(32,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=2)
        return prob, lstm_hidden
    
    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask,prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                         torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                         torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s,a,r,s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]
        
    def train_net(self):
        s,a,r,s_prime,done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden  = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(K_epoch):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()
            
            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()
            
def evaluate_model(model, env, eval_episodes=10):
    total_rewards = []
    for _ in range(eval_episodes):
        h_out = (torch.zeros([1, 1, 32], dtype=torch.float), 
                 torch.zeros([1, 1, 32], dtype=torch.float))
        s, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0

        with torch.no_grad():  # 禁用梯度计算
            while not (done or truncated):
                h_in = h_out
                prob, h_out = model.pi(torch.from_numpy(s).float(), h_in)
                prob = prob.view(-1)
                a = torch.argmax(prob).item()  # 评估时选择概率最大的动作
                s_prime, r, done, truncated, _ = env.step(a)
                episode_reward += r
                s = s_prime

        total_rewards.append(episode_reward)
    return np.mean(total_rewards)      
def main(conf):
    run_name =  f"PPO {conf.seed} {time.strftime('%H:%M %m-%d')}"
    wandb.init(reinit=True, project="Gym-ppo-cartpole", name=run_name, config=OmegaConf.to_container(conf))

    torch.manual_seed(conf.seed)
    env = gym.make('CartPole-v1')
    model = PPO()
    score = 0.0
    print_interval = 10
    eval_episodes = 10       # 每次评估运行10个episode取平均
    best_score = 0
    eval_step = 0
    for n_epi in tqdm(range(1000)):
        h_out = (torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))
        s, _ = env.reset()
        done = False
        truncated = False  # 新增truncated变量
        while not (done or truncated):
            for t in range(T_horizon):
                h_in = h_out
                prob, h_out = model.pi(torch.from_numpy(s).float(), h_in)
                prob = prob.view(-1)
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)

                # 正确处理终止条件
                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), h_in, h_out, done or truncated))  # 存储时标记终止
                s = s_prime

                score += r
                # 同时检查done和truncated
                if done or truncated:
                    break
                    
            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            avg_score = score/print_interval
            # print("# of episode :{}, avg score : {:.1f}".format(n_epi, avg_score))
            # wandb.log({"avg_score": avg_score})
            score = 0.0

        # # 评估阶段
        # if n_epi % eval_interval == 0 and n_epi != 0:
            eval_score = evaluate_model(model, env, eval_episodes)
            # print(f"---- Evaluation after Episode {n_epi}, Avg Eval Score: {eval_score:.1f} ----")
            wandb.log({"avg_score": avg_score,"eval_score": eval_score,"episode_step":n_epi},step=eval_step)
            eval_step += 1
            if best_score < eval_score:
                best_score = eval_score
    env.close()
    return best_score
def sweep(seed: int, conf_override: OmegaConf):
    def _objective(trial: optuna.Trial):
        conf = OmegaConf.merge(conf_override, {
            # "seed": seed * 1000 + trial.number,
            "seed": 16 + trial.number,

            # "project_name": f"E-SNN-sweep",

            # "es_conf": {
            #     "lr":           trial.suggest_categorical("lr",  [0.01, 0.05, 0.1, 0.15, 0.2]),
            #     "eps":          trial.suggest_categorical("eps", [1e-4, 1e-3, 0.01, 0.1, 0.2]),
            # },
            # "network_conf": {
            #     "num_neurons":  trial.suggest_categorical("num_neurons", [128, 256]),
            # }
        })

        metrics = main(conf)
        return metrics

    optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=seed)).optimize(_objective)


if __name__ == "__main__":
    _config = OmegaConf.from_cli()
    if hasattr(_config, "sweep"):
        sweep(_config.sweep, _config)
    else:
        main(_config)
