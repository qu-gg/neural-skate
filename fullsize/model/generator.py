import torch
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt
import numpy as np


class Generator(nn.Module):
    """
    Generator portion of the model.
    CNN of Transpose Convolutions to get to the desired size and shape
    """
    def __init__(self):
        super(Generator, self).__init__()
        # self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        
        self.fc    = nn.Linear(100, 1600)
        self.conv1 = nn.ConvTranspose2d(100, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 100, 4, 4)
        x = f.relu(self.bn1(self.conv1(x)))
        x = f.relu(self.bn2(self.conv2(x)))
        x = f.relu(self.bn3(self.conv3(x)))
        x = f.relu(self.bn4(self.conv4(x)))
        x = torch.tanh(self.conv5(x))
        #
        # x = f.relu(self.conv1(x))
        # x = f.relu(self.conv2(x))
        # x = f.relu(self.conv3(x))
        # x = self.conv4(x)

        return x


def fake_batch(gen, size, show=False):
    """
    Handles creating a batch of generated images
    :param gen: Generator class
    :param size: Number of images to generate
    :param show: Flag to display the first generated image
    :return: Torch tensor of generated images
    """
    noise = torch.Tensor(np.random.uniform(-1.0, 1.0, (size, 100)))
    images = gen(noise)
    labels = [np.random.uniform(0.9, 1.0) for _ in range(size)]

    if show:
        images = images.detach().numpy()

        for i in range(len(images)):
            numpy_images = np.swapaxes(images, 1, 2)
            numpy_images = np.swapaxes(numpy_images, 2, 3)
            plt.imshow(numpy_images[0])
            plt.show()

    labels = torch.Tensor(labels).view([-1, 1])
    return images, labels


if __name__ == '__main__':
    fake_batch(Generator(), 1, True)

