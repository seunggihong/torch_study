import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4,10)
        self.fc2 = nn.Linear(10,3)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))