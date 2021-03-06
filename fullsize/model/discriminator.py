import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
import imageio
from fullsize import utils

img_size = 128
num_color = 3
dataset = "128-set"
cur_idx = 0


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
        self.conv5 = nn.Conv2d(32, 32, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=2, stride=2)
        self.final = nn.Linear(16, 1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(16)

    def forward(self, x):
        # x = f.leaky_relu(self.conv1(x), 0.2)
        # x = f.leaky_relu(self.conv2(x), 0.2)
        # x = f.leaky_relu(self.conv3(x), 0.2)
        # x = f.leaky_relu(self.conv4(x), 0.2)
        # x = f.leaky_relu(self.conv5(x), 0.2)
        # x = f.leaky_relu(self.conv6(x), 0.2)

        x = f.leaky_relu(self.conv1(x), 0.2)
        x = f.leaky_relu(self.bn1(self.conv2(x)), 0.2)
        x = f.leaky_relu(self.bn2(self.conv3(x)), 0.2)
        x = f.leaky_relu(self.bn3(self.conv4(x)), 0.2)
        x = f.leaky_relu(self.bn4(self.conv5(x)), 0.2)
        x = f.leaky_relu(self.bn5(self.conv6(x)), 0.2)

        x = x.view(-1, 16)
        x = self.final(x)
        # print("Output of discrim: ", x.shape)
        return x


def get_indices(size):
    """
    Gets the indices of the next batch of training data, wrapping around if over the end of the data set
    :param size: number of indices to grab
    :return: list of indices
    """
    global cur_idx

    indices = []
    for idx in range(cur_idx, cur_idx + size):
        if idx + size > 3009:
            indices.append(idx % 3009)
        else:
            indices.append(idx)

    cur_idx += size
    if cur_idx > 3009:
        cur_idx %= 3009

    return indices


def real_batch(size, show=False):
    """
    Handles grabbing a random batch of images from the dataset
    :param size: how many images to grab
    :param show: whether to display the first pulled image
    :return: Torch Tensor of image arrays
    """
    random_batch = get_indices(size)
    image_batch = []

    # Reads and preprocesses sampled images
    for number in random_batch:
        image = imageio.imread("../data/{}/{}.jpg".format(dataset, number))
        image = image / 255

        # Rescale to -1 1 for the image
        image = utils.rescale(image, (-1, 1))
        image = image.astype('float32')
        image_batch.append(image)

    # Putting images into torch format
    numpy_images = np.swapaxes(image_batch, 2, 3)
    numpy_images = np.swapaxes(numpy_images, 1, 2)
    images = torch.from_numpy(numpy_images).float().cuda()

    # Sampling labels for batch
    labels = [np.random.uniform(0.0, 0.1) for _ in range(size)]

    if show:
        image = images[0].cpu().detach().numpy()
        image = np.swapaxes(image, 0, 1)
        image = np.swapaxes(image, 1, 2)

        print(image.shape)
        image = np.reshape(image, [img_size, img_size, num_color])
        print(image)

        image = utils.rescale(image, (0, 1))
        plt.imshow(image)
        plt.show()

    labels = torch.Tensor(labels).view([-1, 1])
    return images, labels


if __name__ == '__main__':
    for _ in range(150):
        get_indices(32)
    real_batch(5, True)

