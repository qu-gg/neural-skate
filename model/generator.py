import torch
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt
import numpy as np

img_size = 64


class Generator(nn.Module):
    """
    Generator portion of the model.
    CNN of Transpose Convolutions to get to the desired size and shape
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_1 = nn.ConvTranspose2d(100, 64, kernel_size=5, stride=1)
        self.conv_2 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2)
        self.conv_3 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2)
        self.conv_4 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2)

    def forward(self, x):
        x = x.view(-1, 100, 1, 1)
        x = self.conv_1(x)
        x = f.leaky_relu(self.conv_2(x))
        x = f.leaky_relu(self.conv_3(x))
        x = torch.tanh(self.conv_4(x))
        return x


def fake_batch(gen, size, show=False):
    """
    Handles creating a batch of generated images
    :param gen: Generator class
    :param size: Number of images to generate
    :param show: Flag to display the first generated image
    :return: Torch tensor of generated images
    """
    noise = torch.Tensor(np.random.uniform(-1.0, 1.0, (size, 1600)))
    images = gen(noise)
    labels = [np.random.uniform(0.9, 1.0) for _ in range(size)]

    if show:
        image = images[0].view(img_size, img_size)
        image = image.detach().numpy()
        plt.imshow(image)
        plt.show()

    return images, torch.Tensor(labels)
