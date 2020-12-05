"""
@file latentwalk.py
@author Ryan Missel

Handles performing a latent walk down the dimensions of the GAN to see how it learns the distribution
of the dataset
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import scipy.interpolate as intp
import argparse

import fullsize.utils as utils
from fullsize.model.generator import *
from fullsize.model.discriminator import *

torch.set_default_tensor_type('torch.FloatTensor')

parser = argparse.ArgumentParser(description="Hyper-parameters for network.")
parser.add_argument('-n', '--number', action='store', type=int, default=62000)
args = parser.parse_args()

gen = Generator()
gen.load_state_dict(torch.load('testing/checkpoints/generator{}.ckpt'.format(args.number)))
#
# dis = Discriminator()
# dis.load_state_dict(torch.load('fullsize/testing/checkpoints/discriminator{}.ckpt'.format(args.number)))


# Generate noise matrix based on walk over each dimension (-1, 1)

# Iterate over each interpolation and get the corresponding output
stacks = []

for _ in range(5):
    interp = intp.interp1d([1, 15], np.vstack([np.random.uniform(-1, 1, [1, 100]),
                                               np.random.uniform(-1, 1, [1, 100])]), axis=0)

    vecs = [interp(i) for i in range(1, 15)]
    vecs = torch.from_numpy(np.stack(vecs)).float()

    images = gen(vecs)

    result = np.ones([img_size, img_size, num_color])
    for image in images:
        image = np.swapaxes(image.detach().numpy(), 0, 1)
        image = np.swapaxes(image, 1, 2)
        image = utils.rescale(image, (0, 1))

        result = np.concatenate((result, image), axis=1)

    stacks.append(result)


stacks = np.vstack(stacks)
plt.imshow(stacks)
plt.show()