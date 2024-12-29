import os
import numpy as np
import torch
import torch.optim as optim
import random 

from typing import Literal

from .networks import DQNDNN, DQNCNN

class DQNAgent:
    '''def __init__(self,
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
        
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)'''
    #--------------------------------------------------------------------------
    def __init__(self,
                 input_dim, 
                 n_actions: int,
                 lr: float = 0.01,
                 buffer_size=10000) -> None:
        # Store the number of actions
        self.n_actions = n_actions

        # Ensure input_dim is treated consistently
        if isinstance(input_dim, int):  
            self.use_dnn = True
            input_dim = input_dim  # It's already scalar, no further action needed
        elif isinstance(input_dim, (list, tuple)):
            self.use_dnn = len(input_dim) == 1
            input_dim = input_dim[0] if len(input_dim) == 1 else input_dim
        else:
            raise ValueError(f"Invalid type for input_dim: {type(input_dim)}")

        # Initialize networks
        if self.use_dnn:
            self.eval_net = DQNDNN(input_dim, n_actions)
            self.target_net = DQNDNN(input_dim, n_actions)
        else:
            self.eval_net = DQNCNN(input_dim, n_actions)
            self.target_net = DQNCNN(input_dim, n_actions)

        # Optimizer and replay buffer
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
        self.memory = []  # Replay buffer
        self.buffer_size = buffer_size

    # Sample an action based on epsilon-greedy policy
    def sample(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        if np.random.rand() < epsilon:  # Explore
            #print(np.random.choice(range(self.n_actions)))
            return int(np.random.choice(range(self.n_actions)))
        else:  # Exploit
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.eval_net(state_tensor)
            action = torch.argmax(q_values).item()
            #print(f"Exploiting: Q-Values {q_values.numpy()}, Action {action}")
            return action


    def store_transition(self, state, action, reward, next_state):
        transition = (state, action, reward, next_state)
        if len(self.memory) >= self.buffer_size:
            self.memory.pop(0)
        self.memory.append(transition)

    def learn(self, batch_size=32, gamma=0.99) -> None:
        if len(self.memory) < batch_size:
            return  # Not enough samples to train

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)

        # 從記憶中隨機取樣
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)

        # 將 list 轉換為單一 numpy.ndarray，再轉換為 tensor
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)

        # Compute Q-values
        q_eval = self.eval_net(states).gather(1, actions)

        # Compute target Q-values
        with torch.no_grad():
            q_next = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            q_target = rewards + gamma * q_next

        # Compute loss and update the network
        loss = torch.nn.functional.mse_loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 每隔一定步驟更新 target 網路
        if not hasattr(self, "update_step"):
            self.update_step = 0
        self.update_step += 1
        if self.update_step % 100 == 0:
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
    def update_target_network(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())