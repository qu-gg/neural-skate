import argparse
import os
import torch.optim as optim
from model.discriminator import *
from model.generator import *

parser = argparse.ArgumentParser(description="Hyper-parameters for network.")
parser.add_argument('-b', '--batch', action='store', type=int, default=32)
parser.add_argument('-g', '--gpu', action='store', type=bool, default=True)
parser.add_argument('-s', '--steps', action='store', type=int, default=5000)
parser.add_argument('-n', '--gen', action='store', type=int, default=1)
parser.add_argument('-d', '--dis', action='store', type=int, default=1)
parser.add_argument('-r', '--results', action='store', type=int, default=0)
args = parser.parse_args()

BATCH_SIZE = args.batch
PATH = "testing/results{}".format(args.results)

# Creating paths if they do not exist
paths = ["testing", PATH, "testing/checkpoints"]
for path in paths:
    if not os.path.exists(path):
        os.mkdir(path)

# Writing the hyperparameters to a file
file = open(PATH + "hyperparameters.txt", "w")
file.write("{}".format(args))
file.close()

# Utilizing GPU if available
if torch.cuda.is_available() and args.gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

gen = Generator()  # Initializing nets
dis = Discriminator()

loss = nn.BCELoss()  # Loss function

beta1 = 0.5
beta2 = 0.95
gen_optim = optim.Adam(gen.parameters(), lr=.0001, betas=[beta1, beta2])  # Optimizers
dis_optim = optim.Adam(dis.parameters(), lr=.0001, betas=[beta1, beta2])


def graph(num_iter, d_loss, g_loss):
    """
    Matplotlib function to display the loss per iteration of the model
    :param num_iter: num iterations
    :param d_loss: list of discriminator losses
    :param g_loss: list of generator losses
    """
    plt.plot(num_iter, d_loss, '-b', label='Dis Loss')
    plt.plot(num_iter, g_loss, '-r', label='Gen Loss')
    plt.xlabel("Number of Iterations")
    plt.ylim([0.0, 5.0])
    plt.legend(loc='upper right')
    plt.savefig("testing/results{}/graph_of_loss.png".format(args.results))


def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img


def detach(to_detach):
    """ Function to abstract converting the tensor into a numpy array """
    return to_detach.cpu().detach().numpy()


def training(num_steps):
    """
    Handles the dual training of the Discriminator and Generator
    Trains the Discriminator first on the real images then fake images and backprops
    Then trains the generator on the negative loss from the Discriminator
    :param num_steps: Number of training steps per epoch
    :return: None
    """
    losses = [[], []]

    for step in range(num_steps):
        for _ in range(args.dis):
            """ Discriminator Training """
            dis.zero_grad()

            images, labels = real_batch(BATCH_SIZE)
            fake_images, fake_labels = fake_batch(gen, BATCH_SIZE)

            # Testing discriminator on real and fake images
            dis_real = loss(dis(images), labels)
            dis_fake = loss(dis(fake_images.detach()), fake_labels)

            dis_loss = (dis_real + dis_fake)
            dis_loss.backward()
            dis_optim.step()

        for _ in range(args.gen):
            """ Generator Training """
            gen.zero_grad()
            dis.zero_grad()

            fake_images, fake_labels = fake_batch(gen, BATCH_SIZE)

            # Running fake images through discrim to get generator loss
            gen_loss = loss(dis(fake_images), fake_labels)
            gen_loss.backward()
            gen_optim.step()

        """ Appending losses for plt """
        losses[0].append(detach(dis_loss).item(0))
        losses[1].append(detach(gen_loss).item(0))

        """ Print updated losses and save checkpoints/image at certain intervals """
        if step % 500 == 0:
            print("Dis Loss on {}: {}".format(step, detach(dis_loss)))
            print("Gen Loss on {}: {}".format(step, detach(gen_loss)))

            # Saving image
            images = fake_images.detach().numpy()
            result = np.ones([img_size, img_size, num_color])
            for i in range(5):
                numpy_images = np.swapaxes(images, 1, 2)
                numpy_images = np.swapaxes(numpy_images, 2, 3)
                result = np.concatenate((result, numpy_images), axis=1)
                result = np.concatenate((result, np.ones([img_size, img_size, num_color])), axis=1)
            imageio.imsave("testing/results{}/{}step.png".format(args.results, step), result)

            # Saving current loss images
            steps = [i for i in range(step + 1)]
            graph(steps, losses[0], losses[1])

        if step % 1000 == 0:
            torch.save(gen.state_dict(), "testing/checkpoints/generator{}.ckpt".format(step))
            torch.save(dis.state_dict(), "testing/checkpoints/discriminator{}.ckpt".format(step))


training(args.steps)
