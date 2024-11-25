import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyGradientCNN(nn.Module):
    def __init__(self, 
                 input_dim,
                 n_actions: int) -> None:
        super().__init__()

        width, height = input_dim[1], input_dim[2]
        print('width', width, 'height', height)
        self.conv1 = nn.Conv2d(input_dim[0], 16, kernel_size=3, stride=2, padding=1)
        width, height = (width - 1) // 2 + 1, (height - 1) // 2 + 1
        self.pool1 = nn.MaxPool2d(2)
        width, height = width // 2, height // 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        width, height = (width - 1) // 2 + 1, (height - 1) // 2 + 1
        self.pool2 = nn.MaxPool2d(2)
        width, height = width // 2, height // 2
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # width, height = (width - 1) // 2 + 1, (height - 1) // 2 + 1
        # self.pool3 = nn.MaxPool2d(2)
        # width, height = width // 2, height // 2
        print('width', width, 'height', height)
        self.fc1 = nn.Linear(32 * width * height, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # x = F.relu(self.conv3(x))
        # x = self.pool3(x)
        x = torch.flatten(x, start_dim=0)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)