import torch
import torch.nn as nn
import torch.functional as f
import torch.optim as optim
import numpy


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

    def forward(self, x):
        pass


def get_fake_batch(size):
    pass