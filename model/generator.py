import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


class Generator(nn.Module):
    """
    Generator portion of the model.
    CNN of Transpose Convolutions to get to the desired size and shape
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.dense = nn.Linear(100, 25600)
        self.convo1 = nn.Conv2d(100, 512, kernel_size=5, stride=1, padding=2)
        self.trans1 = nn.ConvTranspose2d(self.convo1.out_channels, self.convo1.out_channels//2, kernel_size=4, stride=4)
        self.convo2 = nn.Conv2d(self.trans1.out_channels, self.trans1.out_channels//2, kernel_size=5, stride=1, padding=2)
        self.trans2 = nn.ConvTranspose2d(self.convo2.out_channels, self.convo2.out_channels//2, kernel_size=4, stride=4)
        self.convo3 = nn.Conv2d(self.trans2.out_channels, self.trans2.out_channels//2, kernel_size=5, stride=1, padding=2)
        self.convo4 = nn.Conv2d(self.convo3.out_channels, 1, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.dense(x)
        x = x.view(-1, 100, 16, 16)
        x = f.leaky_relu(self.convo1(x))
        x = f.leaky_relu(self.trans1(x))
        x = f.leaky_relu(self.convo2(x))
        x = f.leaky_relu(self.trans2(x))
        x = f.leaky_relu(self.convo3(x))
        x = torch.tanh(self.convo4(x))
        return x


def fake_batch(gen, size):
    """
    Handles creating a batch of generated images
    :param gen: Generator class
    :param size: Number of images to generate
    :return: Torch tensor of generated images
    """
    images = []
    for _ in range(size):
        noise = torch.randn(1, 100)
        image = gen(noise)
        images.append(image)

    labels = [np.random.uniform(0.9, 1.0) for _ in range(size)]
    return torch.cat(images), torch.Tensor(labels)