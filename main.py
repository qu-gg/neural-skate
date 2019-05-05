import argparse

import imageio
import torch.optim as optim
from model.discriminator import *
from model.generator import *

parser = argparse.ArgumentParser(description="Hyper-parameters for network.")
parser.add_argument('-b', '--batch', action='store', type=int, default=32)
parser.add_argument('-g', '--gpu', action='store', type=bool, default=False)
parser.add_argument('-s', '--steps', action='store', type=int, default=5000)
parser.add_argument('-n', '--gen_steps', action='store', type=int, default=1)
parser.add_argument('-r', '--results', action='store', type=int, default=0)
args = parser.parse_args()

BATCH_SIZE = args.batch

# Writing the hyperparameters to a file
file = open("testing/results{}/hyperparameters.txt".format(args.results), "w")
file.write("{}".format(args))
file.close()

if torch.cuda.is_available() and args.gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


gen = Generator()       # Initializing nets
dis = Discriminator()

loss = nn.BCELoss()     # Loss function

beta1 = 0.5
beta2 = 0.9
gen_optim = optim.Adam(gen.parameters(), lr=.0004, betas=[beta1, beta2])  # Optimizers
dis_optim = optim.Adam(dis.parameters(), lr=.0004, betas=[beta1, beta2])


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
    plt.show()


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
        """ Discriminator Training """
        images, labels = real_batch(BATCH_SIZE)
        dis_real = dis(images)
        dis_real_l = loss(dis_real, labels)

        images, fake_labels = fake_batch(gen, BATCH_SIZE)
        dis_fake = dis(images)
        dis_fake_l = loss(dis_fake, fake_labels)

        dis_loss = dis_real_l + dis_fake_l
        dis.zero_grad()
        dis_loss.backward()
        dis_optim.step()

        """ Generator Training """
        _, labels = real_batch(BATCH_SIZE)
        images, fake_labels = fake_batch(gen, BATCH_SIZE)
        dis_fake = dis(images)

        gen_loss = loss(dis_fake, labels)

        gen.zero_grad()
        dis.zero_grad()
        gen_loss.backward()
        gen_optim.step()

        """ Appending losses for plt """
        losses[0].append(detach(dis_loss).item(0))
        losses[1].append(detach(gen_loss).item(0))

        """ Print updated losses and save checkpoints/image at certain intervals """
        if step % 1000 == 0:
            print("Dis Loss on {}: {}".format(step, detach(dis_loss)))
            print("Gen Loss on {}: {}".format(step, detach(gen_loss)))

            noise = torch.randn(1, 1600)
            image = detach(gen(noise))
            image = np.reshape(image, (3, 64, 64)).T
            imageio.imwrite("testing/results{}/{}step.jpg".format(args.results, step), image)

        if step % 2000 == 0:
            torch.save(gen.state_dict(), "testing/checkpoints/generator{}.ckpt".format(step))
            torch.save(dis.state_dict(), "testing/checkpoints/discriminator{}.ckpt".format(step))

    """ Plotting and saving the graph """
    steps = [i for i in range(args.steps)]
    graph(steps, losses[0], losses[1])


training(args.steps)
