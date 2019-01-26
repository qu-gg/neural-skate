import torch
import torch.nn as nn
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
        self.pad = nn.ReflectionPad2d(1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv1 = nn.Conv2d(50, 64, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 64,kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = x.view(-1, 50, 4, 4)
        x = self.conv1(self.pad(self.upsample(x)))
        x = self.conv2(self.pad(self.upsample(x)))
        x = self.conv3(self.pad(self.upsample(x)))
        x = self.conv4(self.pad(self.upsample(x)))
        return torch.tanh(x)


def fake_batch(gen, size, show=False):
    """
    Handles creating a batch of generated images
    :param gen: Generator class
    :param size: Number of images to generate
    :param show: Flag to display the first generated image
    :return: Torch tensor of generated images
    """
    noise = torch.Tensor(np.random.uniform(-1.0, 1.0, (size, 800)))
    images = gen(noise)
    labels = [np.random.uniform(0.9, 1.0) for _ in range(size)]

    if show:
        image = images[0].view(3, img_size, img_size)
        image = image.detach().numpy().T
        plt.imshow(image)
        plt.show()

    return images, torch.Tensor(labels)
