"""1D-CNN over 64-step, 3-channel (SINR, BLER, HARQ) radio KPI windows."""
import torch
import torch.nn as nn


class LinkDegradationCNN(nn.Module):
    def __init__(self, in_ch=3, c1=16, c2=64, c3=64, window_len=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, c1, kernel_size=9, padding=4, bias=True)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=9, padding=4, bias=True)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=5, padding=2, bias=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c3, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        logit = self.fc(x).squeeze(-1)
        return logit

    def weight_param_count(self):
        return sum(m.weight.numel() for m in (self.conv1, self.conv2, self.conv3, self.fc))
