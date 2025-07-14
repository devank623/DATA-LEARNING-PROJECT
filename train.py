# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

# Other libraries
import argparse
import os
from os.path import join
import random

# Local imports
from models import Generator, Discriminator
from utils import Average, weights_init


## Parse arguments
parser = argparse.ArgumentParser()
# Files and folders
parser.add_argument('--resume', default=None, help='Checkpoint to resume training from')
# Counts
parser.add_argument('--batch_size', default=128, type=int, help='Training batch size')
parser.add_argument('--save_freq', default=5, type=int, help='Checkpoint save frequency')
parser.add_argument('--num_epochs', default=50, type=int, help='Number of training epochs')
# Models parameters
parser.add_argument('--image_size', default=64, type=int, help='Spatial size of training images')
parser.add_argument('--nc', default=3, type=int, help='Number of channels in the training images')
parser.add_argument('--nz', default=100, type=int, help='Size of generator input')
parser.add_argument('--ngf', default=64, type=int, help='Generator feature maps size')
parser.add_argument('--ndf', default=64, type=int, help='Discriminator feature maps size')
# Optimizers settings
parser.add_argument('--lr', default=0.0002, type=float, help='Learning rate')
parser.add_argument('--beta1', default=0.5, type=float, help='Beta1 hyperparameter for Adam optimizers')
# Others
parser.add_argument('--dataset', help='torchvision dataset to use', required=True)
parser.add_argument('--seed', default=None, type=int, help='Set seed for reproducibility, if not set it will be randomized')
args = parser.parse_args()


# If the seed has not been set, randomize it
if args.seed is None:
    args.seed = random.randint(1, 10000)

# Set seed
print('Seed: {}'.format(args.seed))
random.seed(args.seed)
torch.manual_seed(args.seed)


# Create output folder, if not existent
if not os.path.exists(join('checkpoints', args.dataset)):
    os.makedirs(join('checkpoints', args.dataset))


# Create the dataset and dataloader
transform = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * args.nc, (0.5,) * args.nc)
])
dataset_class = getattr(dset, args.dataset)
dataset = dataset_class('./data', transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Create the generator
generator = Generator(args.nz, args.ngf, args.nc).to(device)
generator.apply(weights_init)  # Randomly initialize all weights to mean=0, stdev=0.2
print(generator)

# Create the discriminator
discriminator = Discriminator(args.nc, args.ndf).to(device)
discriminator.apply(weights_init)  # Randomly initialize all weights to mean=0, stdev=0.2
print(discriminator)


# Initialize BCELoss function
criterion = nn.BCELoss()

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both generator and discriminator
optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))


# Resume training
starting_epoch = 1
if args.resume is not None:
    print('Resuming training from epoch {}'.format(args.resume))
    checkpoint = torch.load(args.resume)

    # Load models
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    # Load optimizers
    optimizer_G.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizerD_state_dict'])

    starting_epoch = checkpoint['epoch'] + 1


# Start training
print('Training models')
generator.train()
discriminator.train()

# Training loop
for epoch in range(starting_epoch, args.num_epochs + 1):

    # Keep track of average epoch losses and metrics
    avg_G_loss = Average()
    avg_D_loss = Average()
    avg_D_x = Average()
    avg_D_G_z1 = Average()
    avg_D_G_z2 = Average()

    # For each batch in the dataloader
    for data in dataloader:

        ## Train discriminator with real data
        discriminator.zero_grad()

        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        # Forward + backward
        output = discriminator(real_cpu).view(-1)
        err_D_real = criterion(output, label)
        err_D_real.backward()
        D_x = output.mean().item()

        ## Train discriminator with generated data
        # Generate batch of latent vectors and use it to generate fake images
        noise = torch.randn(b_size, args.nz, 1, 1, device=device)
        fake = generator(noise)
        label.fill_(fake_label)

        # Forward + backward
        output = discriminator(fake.detach()).view(-1)
        err_D_fake = criterion(output, label)
        err_D_fake.backward()
        D_G_z1 = output.mean().item()

        # Compute loss of discriminator as sum of losses and optimize
        err_D = err_D_real + err_D_fake
        optimizer_D.step()

        ## Update generator
        generator.zero_grad()
        label.fill_(real_label)  # The generator loss takes also into consideration fake labels

        # Perform another forward pass of the fake batch through the updated discriminator
        output = discriminator(fake).view(-1)
        err_G = criterion(output, label)
        err_G.backward()
        D_G_z2 = output.mean().item()

        # Optimize generator
        optimizer_G.step()

        # Update batch averages
        avg_G_loss.update(err_G.item())
        avg_D_loss.update(err_D.item())
        avg_D_x.update(D_x)
        avg_D_G_z1.update(D_G_z1)
        avg_D_G_z2.update(D_G_z2)

    # Output training stats
    print('Epoch {}/{} - Average batch losses and metrics\n'
        '    Generator loss:     {}\n'
        '    Discriminator loss: {}\n'
        '    D(x):               {}\n'
        '    D(G(z1)):           {}\n'
        '    D(G(z2)):           {}'.format(
        epoch, args.num_epochs,
        avg_G_loss.mean(), avg_D_loss.mean(),
        avg_D_x.mean(), avg_D_G_z1.mean(), avg_D_G_z2.mean()
    ))

    # Save models
    if epoch % args.save_freq == 0:
        print('Saving models')
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizerG_state_dict': optimizer_G.state_dict(),
            'optimizerD_state_dict': optimizer_D.state_dict()
        }, join('checkpoints', args.dataset, 'epoch_{}.pt'.format(epoch)))
