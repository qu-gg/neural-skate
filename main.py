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


beta1 = 0.0
beta2 = 0.9
gen_optim = optim.Adam(gen.parameters(), lr=.0001, betas=[beta1, beta2])  # Optimizers
dis_optim = optim.Adam(dis.parameters(), lr=.0001, betas=[beta1, beta2])


def detach(to_detach):
    """ Function to abstract converting the tensor into a numpy array """
    return to_detach.cpu().detach().numpy()


def training(num_epochs, num_steps):
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
        for step in range(num_steps):
            """ Discriminator Training """
            dis_optim.zero_grad()

            images, labels = real_batch(BATCH_SIZE)
            dis_real = dis(images)
            dis_real_l = loss(dis_real, labels)

            images, fake_labels = fake_batch(gen, BATCH_SIZE)
            dis_fake = dis(images)
            dis_fake_l = loss(dis_fake, fake_labels)

            dis_loss = dis_real_l + dis_fake_l
            dis_loss.backward()
            dis_optim.step()

            """ Generator Training """
            for _ in range(10):
                gen_optim.zero_grad()

                images, fake_labels = fake_batch(gen, BATCH_SIZE)
                dis_fake = dis(images)

                gen_loss = loss(dis_fake, fake_labels)
                gen_loss.backward()

                gen_optim.step()

            if step % 10 == 0:
                print("Dis Loss on {}: {}".format(step, detach(dis_loss)))
                print("Gen Loss on {}: {}".format(step, detach(gen_loss)))

                noise = torch.randn(1, 100)
                image = detach(gen(noise))
                image = np.reshape(image, (256, 256))
                misc.imsave("testing/results/{}epoch.jpg".format(epoch), image)

                PATH = "testing/checkpoints/"
                torch.save(gen.state_dict(), PATH + "gen_epoch{}.ckpt".format(epoch))
                torch.save(dis.state_dict(), PATH + "dis_epoch{}.ckpt".format(epoch))


epochs = int(input("Number of epochs: "))
steps = int(input("Steps per epoch: "))
training(epochs, steps)
