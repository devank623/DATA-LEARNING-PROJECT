# Torch imports
import torch
import torchvision.utils as vutils

# Other libraries
import argparse
import os
from os.path import join
import random

# Add parent directory to path
import sys
sys.path.insert(0, '..')

# Local imports
from models import Generator


## Parse arguments
parser = argparse.ArgumentParser()
# Files and folders
parser.add_argument('--checkpoint', help='Path to checkpoint', required=True)
parser.add_argument('--name', default='interpolation', help='Name of the output file')
# Counts
parser.add_argument('--steps', default=8, type=int, help='Steps to interpolate between the two points in the latent space')
# Models parameters
parser.add_argument('--nc', default=3, type=int, help='Number of channels in the training images')
parser.add_argument('--nz', default=100, type=int, help='Size of generator input')
parser.add_argument('--ngf', default=64, type=int, help='Generator feature maps size')
# Others
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
if not os.path.exists('output'):
    os.makedirs('output')


# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Load checkpoint
generator = Generator(args.nz, args.ngf, args.nc).to(device)
checkpoint = torch.load(args.checkpoint)
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()


def interpolate(i1, i2, steps=10):
    ratios = torch.linspace(0, 1, steps=steps)

    # Linearly interpolate vectors
    vectors = torch.empty((steps, *i1.shape))
    for i, r in enumerate(ratios):
        vectors[i] = (1 - r) * i1 + r * i2

    return vectors


# Generate two latent points
noise = torch.randn(2, args.nz, 1, 1, device=device)

# Interpolate between the two points
interpolated = interpolate(noise[0], noise[1], steps=args.steps)
interpolated = interpolated.to(device)

# Generate images
with torch.no_grad():
    fake = generator(interpolated)
    fake = (fake + 1) * 0.5
    vutils.save_image(fake, join('./output', '{}.jpg'.format(args.name)))
