import torch
import torch.nn as nn
import torch.nn.functional as f
import random as r
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def real_batch(size, show=False):
    """
    Handles grabbing a random batch of images from the dataset
    :param size: how many images to grab
    :param show: whether to display the first pulled image
    :return: Torch Tensor of image arrays
    """
    random_batch = [r.randint(0, 3009) for _ in range(size)]
    image_batch = []

    for number in random_batch:
        image_path = misc.imread("dataset/decks/" + str(number) + ".jpg")
        image = torch.Tensor(image_path)
        image_batch.append(image)

    image_batch = torch.cat(image_batch)
    image_batch = image_batch.view(-1, 3, 256, 256)
    classes = [np.random.uniform(0.0, 0.1) for _ in range(size)]

    if show:
        plt.imshow(image_batch[0])
        plt.show()
    return image_batch, torch.Tensor(classes)


class Discriminator(nn.Module):
    """
    Discriminator portion of the model
    Basic CNN outputting a number between 0 and 1
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        size = 32
        self.conv1 = nn.Conv2d(3, size, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, self.conv1.out_channels*2, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels*2, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(self.conv3.out_channels, self.conv3.out_channels*2, kernel_size=5, stride=2)
        self.conv5 = nn.Conv2d(self.conv4.out_channels, self.conv4.out_channels*2, kernel_size=5, stride=2)
        self.conv6 = nn.Conv2d(self.conv5.out_channels, self.conv5.out_channels, kernel_size=5, stride=2)
        self.drop = nn.Dropout(0.4)
        self.final = nn.Linear(512, 1)

    def forward(self, x):
        x = f.leaky_relu(self.conv1(x))
        x = f.leaky_relu(self.conv2(x))
        x = f.leaky_relu(self.conv3(x))
        x = f.leaky_relu(self.conv4(x))
        x = f.leaky_relu(self.conv5(x))
        x = f.leaky_relu(self.conv6(x))
        x = self.drop(x)
        x = x.view(-1, 512)
        x = torch.sigmoid(self.final(x))
        return x