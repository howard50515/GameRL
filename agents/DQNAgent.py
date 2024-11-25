import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random
from typing import Literal

from .networks import DQNDNN, DQNCNN

class DQNAgent:
    def __init__(self,
                 input_dim, 
                 n_actions: int,
                 lr: float = 0.01) -> None:
        self.use_dnn = len(input_dim) == 1
        input_dim = input_dim[0] if len(input_dim) == 1 else input_dim
        
        if self.use_dnn:
            self.eval_net = DQNDNN(input_dim, n_actions)
            self.target_net = DQNDNN(input_dim, n_actions)
        else:
            self.eval_net = DQNCNN(input_dim, n_actions)
            self.target_net = DQNCNN(input_dim, n_actions)
        
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
        self.memory = []  # 經驗回放緩衝區
        self.memory_size = 10000  # 最大緩衝區大小
        self.batch_size = 32  # 批量大小
        self.gamma = 0.99  # 折扣因子

    def sample(self, 
               state: np.ndarray, 
               epsilon: float = 0.0) -> int:
        action = 0
        if random.random() < epsilon:
            action = random.choice([0, 1])
        else:
            q_values = self.eval_net(torch.tensor(state, dtype=torch.float32))
            action = torch.argmax(q_values).item()
        return action

    def store_transition(self, 
                         state: np.ndarray, 
                         action: Literal[0, 1], 
                         reward: float, 
                         next_state: np.ndarray) -> None:
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)  # 超過容量時刪除最早的經驗
        self.memory.append((state, action, reward, next_state))

    def learn(self) -> None:
        """訓練步驟"""
        # 檢查緩衝區資料是否足夠
        if len(self.memory) < self.batch_size:
            return
        
        # 從緩衝區中隨機抽樣
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        # 將 numpy 轉換為 torch 張量
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        # 計算目標 Q 值
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]  # 最大值
            target_q = rewards + self.gamma * max_next_q_values

        # 計算當前 Q 值
        q_values = self.eval_net(states)
        eval_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 計算損失
        loss = nn.MSELoss()(eval_q, target_q)

        # 更新參數
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 每隔一定步驟更新 target_net
        if hasattr(self, "update_step") and self.update_step % 100 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
    def load(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            print("Checkpoint file not found, use default settings")
            return
        
        checkpoint = torch.load(ckpt_path)
        self.eval_net.load_state_dict(checkpoint['eval_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save(self, ckpt_path):
        torch.save({
            'eval_net_state_dict': self.eval_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, ckpt_path)