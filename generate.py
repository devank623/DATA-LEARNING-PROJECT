# Torch imports
import torch
import torchvision.utils as vutils

# Other libraries
import argparse
from os.path import join
import random

# Local imports
from models import Generator


## Parse arguments
parser = argparse.ArgumentParser()
# Files and folders
parser.add_argument('--checkpoint', help='Path to checkpoint', required=True)
parser.add_argument('--name', default='generation', help='Name of the output file')
# Counts
parser.add_argument('--batch_size', default=32, type=int, help='Number of images to generate')
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


# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Load checkpoint
generator = Generator(args.nz, args.ngf, args.nc).to(device)
checkpoint = torch.load(args.checkpoint)
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()


# Create batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(args.batch_size, args.nz, 1, 1, device=device)

# Generate images
with torch.no_grad():
    fake = generator(fixed_noise)
    fake = (fake + 1) * 0.5
    vutils.save_image(fake, join('./output', '{}.jpg'.format(args.name)))
