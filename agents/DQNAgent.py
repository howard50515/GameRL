import os
import numpy as np
import torch
import torch.optim as optim
import random

from typing import Literal

from .networks import DQNDNN, DQNCNN


from collections import deque

class DQNAgent:
    def __init__(self, input_dim, n_actions: int, lr: float = 0.01, buffer_size=10000, device=None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.memory = deque(maxlen=buffer_size)
        self.use_dnn = isinstance(input_dim, int) or (isinstance(input_dim, (list, tuple)) and len(input_dim) == 1)
        input_dim = input_dim[0] if isinstance(input_dim, (list, tuple)) and len(input_dim) == 1 else input_dim

        self.eval_net = (DQNDNN(input_dim, n_actions) if self.use_dnn else DQNCNN(input_dim, n_actions)).to(self.device)
        self.target_net = (DQNDNN(input_dim, n_actions) if self.use_dnn else DQNCNN(input_dim, n_actions)).to(self.device)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)

    def sample(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.choice(range(self.n_actions)))
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.eval_net(state_tensor)).item()

    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def learn(self, batch_size=32, gamma=0.99) -> None:
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = map(np.array, zip(*batch))
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)

        q_eval = self.eval_net(states).gather(1, actions)
        with torch.no_grad():
            q_target = rewards + gamma * self.target_net(next_states).max(dim=1, keepdim=True)[0]

        loss = torch.nn.functional.mse_loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def load(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            print("Checkpoint file not found, use default settings")
            return

        checkpoint = torch.load(ckpt_path, map_location=self.device)  # 加載時確保模型在正確設備
        self.eval_net.load_state_dict(checkpoint['eval_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save(self, ckpt_path):
        torch.save({
            'eval_net_state_dict': self.eval_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, ckpt_path)

    def update_target_network(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())
