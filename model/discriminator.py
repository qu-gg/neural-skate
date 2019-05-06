import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision.utils import save_image
import random as r
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

img_size = 64
num_color = 3
dataset = "64-set"


class Discriminator(nn.Module):
    """
    Discriminator portion of the model
    Basic CNN outputting a number between 0 and 1
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_color, 64, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=4, stride=2)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=2, stride=2)
        self.final = nn.Linear(16, 1)

    def forward(self, x):
        x = f.leaky_relu(self.conv1(x))
        x = f.leaky_relu(self.conv2(x))
        x = f.leaky_relu(self.conv3(x))
        x = f.leaky_relu(self.conv4(x))
        x = f.leaky_relu(self.conv5(x))
        x = x.view(-1, 16)
        return torch.sigmoid(self.final(x))


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
        image = misc.imread("data/{}/{}.jpg".format(dataset, number))
        image = np.reshape(image, [num_color, img_size, img_size])
        image_batch.append(image)

    numpy_images = np.asarray(image_batch)
    images = torch.from_numpy(numpy_images).float().cuda()

    labels = [np.random.uniform(0.0, 0.1) for _ in range(size)]

    if show:
        image = np.reshape(numpy_images[0], [img_size, img_size, num_color])
        plt.imshow(image)
        plt.show()
    return images, torch.Tensor(labels)
