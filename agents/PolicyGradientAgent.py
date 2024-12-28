import os
import numpy as np
import torch
import torch.optim as optim

from torch.distributions import Categorical

from .networks import PolicyGradientDNN, PolicyGradientCNN

BUFFER_SIZE = 300

class PolicyGradientAgent():
    def __init__(self, 
                 input_dim: tuple[int, ...],
                 n_actions: int,
                 lr: float = 1e-4,
                 gamma: float = 0.95,) -> None:
        
        self.use_dnn = len(input_dim) == 1
        input_dim = input_dim[0] if len(input_dim) == 1 else input_dim

        if self.use_dnn:
            self.network = PolicyGradientDNN(input_dim, n_actions)
        else:
            self.network = PolicyGradientCNN(input_dim, n_actions)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.995)
        self.rewards = [0] * BUFFER_SIZE
        self.log_probs = [0] * BUFFER_SIZE
        self.index = 0
        self.count = 0
        self.episode_rewards = []
        self.episode_log_probs = []
        self.gamma = gamma

    def sample(self, 
               state: np.ndarray, 
               temperature: float = 1.0) -> tuple[int, torch.Tensor]:
        logits = self.network(torch.FloatTensor(state))
        action_prob = torch.softmax(logits / temperature, dim=-1)
        action_dist = Categorical(action_prob)
        action = action_dist.sample()

        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def discounted_cumulative_rewards(self, rewards):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        return discounted_rewards
    
    def store_transition(self,
                         reward: float,
                         log_prob: torch.Tensor) -> None:
        self.rewards[self.index] = reward
        self.log_probs[self.index] = log_prob
        
        self.index = (self.index + 1) % BUFFER_SIZE
        self.count = min(self.count + 1, BUFFER_SIZE)

    def store_episode(self) -> None:
        """
        儲存一個 episode 所累積的 reward 及 log_prob
        """
        if self.count < 300:
            rewards = self.rewards[:self.count]
            log_probs = self.log_probs[:self.count]
        elif self.count == 300:
            rewards = [self.rewards[(self.index + i) % BUFFER_SIZE] for i in range(BUFFER_SIZE)]
            log_probs = [self.log_probs[(self.index + i) % BUFFER_SIZE] for i in range(BUFFER_SIZE)]
        else:
            print("something wrong in rewards and log_probs")

        rewards = self.discounted_cumulative_rewards(rewards)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # 將 reward 正規標準化
        self.episode_rewards.append(rewards)
        self.episode_log_probs.append(log_probs)

        self.index = 0
        self.count = 0

    def learn(self) -> None:
        if self.count > 0:
            self.store_episode()

        self.optimizer.zero_grad()
        for rewards, log_probs in zip(self.episode_rewards, self.episode_log_probs):
            rewards = torch.FloatTensor(rewards)
            log_probs = torch.stack(log_probs)

            loss = (-log_probs * rewards).sum()
            
            loss.backward()

        self.optimizer.step()
        
        self.episode_rewards = []
        self.episode_log_probs = []

    def load(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            print("Checkpoint file not found, use default settings")
            return
        
        checkpoint = torch.load(ckpt_path)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save(self, ckpt_path):
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, ckpt_path)