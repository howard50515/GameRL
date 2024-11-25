import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNDNN(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 n_actions: int) -> None:
        super().__init__()

        # 全連接層
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))  # 使用 ReLU 激活函數
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)  # 輸出 Q 值向量
