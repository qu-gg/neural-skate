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
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv1 = nn.Conv2d(100, 64, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=0)

    def forward(self, x):
        x = x.view(-1, 100, 4, 4)
        x = f.leaky_relu(self.conv1(self.upsample(x)))
        # print(x.shape)
        x = f.leaky_relu(self.conv2(self.upsample(x)))
        # print(x.shape)
        x = f.leaky_relu(self.conv3(self.upsample(x)))
        # print(x.shape)
        x = f.leaky_relu(self.conv4(self.upsample(x)))
        # print(x.shape)
        x = f.leaky_relu(self.conv5(self.upsample(x)))
        # print(x.shape)
        return torch.tanh(x)


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
        images = images.detach().numpy()

        for i in range(len(images)):
            numpy_images = np.swapaxes(images, 1, 2)
            numpy_images = np.swapaxes(numpy_images, 2, 3)
            plt.imshow(numpy_images[0])
            plt.show()

    return images, torch.Tensor(labels)


if __name__ == '__main__':
    fake_batch(Generator(), 1, True)
