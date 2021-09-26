import torch
import torch.nn.functional as F
import torch.nn as nn


class FC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FC, self).__init__()
        self.fc = torch.nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)

class MLP(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super(MLP, self).__init__()
        self.fc1 = FC(in_ch, hid_ch)
        self.fc2 = FC(hid_ch, out_ch)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)




