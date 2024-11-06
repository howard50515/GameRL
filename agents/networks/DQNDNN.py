import torch
import torch.nn as nn

class DQNDNN(nn.Module):
    def __init__(self, 
                 input_dim,
                 n_actions: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc5 = nn.Linear(32, n_actions)

    def forward(self, state):
        hid = torch.sigmoid(self.fc1(state))
        hid = torch.sigmoid(self.fc2(hid))
        hid = torch.sigmoid(self.fc3(hid))
        hid = torch.sigmoid(self.fc4(hid))
        return self.fc5(hid)