import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 1)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def create_model() -> Net:
    return Net()