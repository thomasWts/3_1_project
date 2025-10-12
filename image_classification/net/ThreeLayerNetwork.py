import torch
import torch.nn as nn
import torch.nn.functional as F

class Three_Layer_Network(nn.Module):
    def __init__(self):
        super(Three_Layer_Network, self).__init__()
        self.fc1 = nn.Linear(3*224*224, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = x.reshape(x.size(0), -1) # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
