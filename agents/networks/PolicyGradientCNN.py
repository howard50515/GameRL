import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyGradientCNN(nn.Module):
    def __init__(self, 
                 input_dim,
                 n_actions: int) -> None:
        super().__init__()
        # self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1) # (16, 200, 150)
        self.pool1 = nn.MaxPool2d(2) # (16, 100, 75)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # (32, 50, 38)
        self.pool2 = nn.MaxPool2d(2) # (32, 25, 19)
        self.fc1 = nn.Linear(32 * 25 * 19, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=0)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)