import argparse
import torch.optim as optim
from model.discriminator import *
from model.generator import *

parser = argparse.ArgumentParser(description="Hyper-parameters for network.")
parser.add_argument('-b', '--batch', action='store', type=int, default=32)
parser.add_argument('-g', '--gpu', action='store', type=bool, default=False)
args = parser.parse_args()

BATCH_SIZE = args.batch

if torch.cuda.is_available() and args.gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


gen = Generator()       # Initializing nets
dis = Discriminator()

loss = nn.BCELoss()     # Loss function

gen_optim = optim.Adam(gen.parameters(), lr=.00005)  # Optimizers
dis_optim = optim.SGD(dis.parameters(), lr=.00005)


def detach(to_detach):
    """ Function to abstract converting the tensor into a numpy array """
    return to_detach.cpu().detach().numpy()


def training(num_epochs):
    """
    Handles the dual training of the Discriminator and Generator
    Trains the Discriminator first on the real images then fake images and backprops
    Then trains the generator on the negative loss from the Discriminator
    :param num_epochs: Number of epochs to run through
    :param steps: Number of training steps per epoch
    :return: None
    """
    for epoch in range(num_epochs):
        print("Epoch {}:".format(epoch))

        """ Discriminator Training """
        dis_optim.zero_grad()

        images, labels = real_batch(BATCH_SIZE)
        dis_real = dis(images)

        images, fake_labels = fake_batch(gen, BATCH_SIZE)
        dis_fake = dis(images)

        dis_loss = loss(dis_real - dis_fake, labels)
        dis_loss.backward()
        dis_optim.step()
        print("Dis Loss on {}: {}".format(epoch, detach(dis_loss)))

        """ Generator Training """
        gen_optim.zero_grad()

        images, fake_labels = fake_batch(gen, BATCH_SIZE)
        dis_fake = dis(images)

        gen_loss = loss(dis_fake - dis_real, fake_labels)
        gen_loss.backward()

        gen_optim.step()
        print("Gen Loss on {}: {}".format(epoch, detach(gen_loss)))

        if epoch % 10 == 0:
            images = detach(torch.zeros(256, 256, 3))
            for _ in range(5):
                noise = torch.randn(1, 100)
                image = detach(gen(noise))
                image = np.reshape(image.T, (256, 256, 3))
                images = np.concatenate((image, images), axis=1)
            misc.imsave("results/{}epoch.jpg".format(epoch), images)


epochs = int(input("Number of epochs: "))
training(epochs)
