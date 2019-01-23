"""
    Define a simple logistic model on each session data
    Use a neural network with single layer to mimic the output of
    logistic model

    Use the pytorch modules to create this model
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()

    self.linear = nn.Linear()

    def forward(self, x):
        out = self.linear(x)
        return out
