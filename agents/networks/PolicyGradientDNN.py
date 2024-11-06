import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyGradientDNN(nn.Module):
    def __init__(self, 
                 input_dim,
                 n_actions: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, state):
        hid = torch.relu(self.fc1(state))
        hid = torch.relu(self.fc2(hid))
        return self.fc3(hid)