import os
import numpy as np
import torch
import torch.optim as optim

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

    def sample(self, 
               state: np.ndarray, 
               epsilon: float = 0.0) -> int:
        return 0

    def store_transition(self, 
                         state: np.ndarray, 
                         action: Literal[0, 1], 
                         reward: float, 
                         next_state: np.ndarray) -> None:
        pass

    def learn(self) -> None:
        pass

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