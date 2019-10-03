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

        self.conv1 = nn.Conv2d(100, 64, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=0)

        torch.nn.init.xavier_uniform(self.conv1.weight)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        torch.nn.init.xavier_uniform(self.conv3.weight)
        torch.nn.init.xavier_uniform(self.conv4.weight)

    def forward(self, x):
        print(x.shape)
        x = x.view(-1, 100, 4, 4)
        x = self.conv1(self.pad(self.upsample(x)))
        print(x.shape)
        x = self.conv2(self.pad(self.upsample(x)))
        print(x.shape)
        x = self.conv3(self.pad(self.upsample(x)))
        print(x.shape)
        x = self.conv4(self.pad(self.upsample(x)))
        print(x.shape)
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
        images = images.detach().numpy()

        for i in range(len(images)):
            image = np.reshape(images[i], [img_size, img_size, 3]).clip(0)
            print(image)
            plt.imshow(image)
            plt.show()

    return images, torch.Tensor(labels)


if __name__ == '__main__':
    fake_batch(Generator(), 1, True)
