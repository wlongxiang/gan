import argparse
import csv
import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import datasets
import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=128),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Linear(in_features=512, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Linear(in_features=1024, out_features=784),
            nn.Tanh()  # note that since the training images are normalized to [-1,1], tanh is a natural choice here
            # there are also people suggesting that scale the output to [-1,1] helps with training of GANs
        )

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

    def forward(self, z):
        # Generate images from z
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=784, out_features=512),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )
        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

    def forward(self, img):
        # return discriminator score for img
        return self.model(img)


def generate_and_save_samples(model, file_name, sample_size):
    samples = torch.randn(sample_size).to(device)
    samples = model(samples)
    # lets reshape and shift to [0,1]
    samples = samples.reshape(-1, 1, 28, 28) * 0.5 + 0.5
    arrays = make_grid(samples, nrow=5)[0]
    img_path = os.path.join(os.path.dirname(__file__), 'ganresults', file_name)
    if arrays.is_cuda:
        arrays = arrays.cpu()
    plt.imsave(img_path, arrays.detach().numpy(), cmap="binary")


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    # init logging
    if not os.path.exists("ganresults"):
        os.mkdir("ganresults")
    ts = int(time.time())
    cvs_file = 'ganresults/result_{}.csv'.format(ts)
    cols_data = ['epoch', 'batch', 'gen_loss', 'disc_loss']
    with open(cvs_file, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(cols_data)
    # init some values
    total_batches = len(dataloader)
    avg_loss_discriminator = 0
    avg_loss_generator = 0
    # loss function
    # note the difference between BCELoss and BCEwithLogitsLoss
    # the former needs sigmoid inputs, while the latter doesn't
    binary_cross_entropy_loss = nn.BCELoss()
    for epoch in range(args.n_epochs):

        if epoch == 0 or epoch % args.save_interval == 0:
            fname = 'samples_epoch_{}_{}.png'.format(epoch, ts)
            generate_and_save_samples(generator, fname, (25, args.latent_dim))

        for i, (imgs, _) in enumerate(dataloader):
            # Train Discriminator
            # -------------------
            training_imgs_batch = imgs.reshape(args.batch_size, -1).to(device)
            _seed = torch.randn((args.batch_size, args.latent_dim)).to(device)
            generated_imgs_batch = generator(_seed)
            optimizer_D.zero_grad()
            if training_imgs_batch.shape[1] != 784:
                # in case our batch size is not divisible by dataset size, this will happen
                print("!!!! image shap is {}, not of 784!!".format(imgs.shape[1]))
                break
            preds_training_imgs = discriminator(training_imgs_batch)
            preds_generated_imgs = discriminator(generated_imgs_batch)
            targets_training_imgs = torch.ones((args.batch_size, 1), dtype=torch.float32).to(device)
            targets_generated_imgs = torch.zeros((args.batch_size, 1), dtype=torch.float32).to(device)
            # label smoothing trick, this trick stabling the training
            # the number is is arbitrary
            targets_training_imgs.uniform_(0.8, 1.2)
            # for training images the disriminator target is 1, for fake images the disriminator target is 0
            loss_discminator = binary_cross_entropy_loss(preds_training_imgs, targets_training_imgs) + \
                               binary_cross_entropy_loss(preds_generated_imgs, targets_generated_imgs)

            loss_discminator.backward()
            optimizer_D.step()
            # -------------------

            # Train Generator
            # -------------------
            _seed = torch.randn((args.batch_size, args.latent_dim)).to(device)
            generated_imgs_batch = generator(_seed)
            optimizer_G.zero_grad()
            preds_generated_imgs = discriminator(generated_imgs_batch)
            # the loss of generator is to mimick the training images
            loss_g = binary_cross_entropy_loss(preds_generated_imgs, targets_training_imgs)
            loss_g.backward()
            optimizer_G.step()
            # -------------------

            avg_loss_discriminator += loss_discminator.item() / args.print_interval
            avg_loss_generator += loss_g.item() / args.print_interval
            batch_num = i + 1
            if batch_num % args.print_interval == 0:
                print('epoch {}/{} batch {}/{} loss_discriminator: {:.3f} loss_generator: {:.3f}'.format(
                    epoch, args.n_epochs,
                    batch_num,
                    total_batches,
                    avg_loss_discriminator,
                    avg_loss_generator))
                csv_data = [epoch, i + 1, avg_loss_generator, avg_loss_discriminator]
                with open(cvs_file, 'a') as fd:
                    writer = csv.writer(fd)
                    writer.writerow(csv_data)

                avg_loss_discriminator = 0
                avg_loss_generator = 0


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = DataLoader(datasets.MNIST('./data/mnist', train=True, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5,), (0.5,))])),
                            batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(latent_dim=args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    # there are 60k inputs in MNIST, try to set the batch size to be divisible, otherwise there are strange batch at
    # end of epoch
    parser.add_argument('--batch_size', type=int, default=60,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--print_interval', type=int, default=50,
                        help='log every print_interval iterations')
    args = parser.parse_args()

    main()
