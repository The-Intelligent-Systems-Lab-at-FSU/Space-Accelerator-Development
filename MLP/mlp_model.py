import torch
from torch import nn
import torch.nn.functional as F


class Classifier(torch.nn.Module):
    """Docstring"""

    def __init__(self):
        """Docstring"""
        super().__init__()

        self.fc1 = nn.Linear(4096 * 3, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, 5)

    def forward(self, x):
        """Docstring"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))

        return F.log_softmax(self.fc7(x), dim=1)
