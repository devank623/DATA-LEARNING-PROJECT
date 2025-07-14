# Torch imports
import torch
import torchvision.utils as vutils

# Other libraries
import argparse
import matplotlib.pyplot as plt
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
parser.add_argument('--name', default='arithmetic', help='Name of the output file')
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


def choose():
    print('We will be doing vector arithmetic A - B + C')
    ABC = dict()

    with torch.no_grad():
        while len(ABC) != 3:

            # Choose vector
            key = None
            while key not in ['A', 'B', 'C']:
                key = input('Choose three images for: ')
            ABC[key] = []

            # Choose three images
            while len(ABC[key]) != 3:

                # Generate some images
                n_images = 10
                noise = torch.randn(n_images, args.nz, 1, 1, device=device)
                fake = generator(noise)
                fake = (fake + 1) * 0.5
                fake = fake.permute(0, 2, 3, 1)

                # Show them
                plt.figure(figsize=(n_images * 1.5, 2))
                for i in range(n_images):
                    plt.subplot(1, n_images, i + 1)
                    plt.imshow(fake[i].cpu())
                    plt.axis('off')
                    plt.title(str(i))
                plt.show()
                plt.close()

                # Input choice
                idx = int(input('Choose an image (-1 for new images): '))
                if 0 <= idx < n_images:
                    ABC[key].append(noise[idx])

    return ABC


ABC = choose()

# Save vectors
with torch.no_grad():
    for vn in ['A', 'B', 'C']:
        vector = torch.stack(ABC[vn], dim=0)
        fake = generator(vector)
        fake = (fake + 1) * 0.5
        vutils.save_image(fake, join('./output', '{}.jpg'.format(vn)))

# Average three images chosen for each vector
A = ABC['A']
B = ABC['B']
C = ABC['C']
A_avg = (A[0] + A[1] + A[2]) / 3
B_avg = (B[0] + B[1] + B[2]) / 3
C_avg = (C[0] + C[1] + C[2]) / 3

# Place noise vectors in mini batch
final_noise = A_avg - B_avg + C_avg
noise = torch.stack([A_avg, B_avg, C_avg, final_noise], dim=0)

# Generate images and show them
with torch.no_grad():
    fake = generator(noise)
    fake = (fake + 1) * 0.5
    vutils.save_image(fake, join('./output', 'a_avg-b_avg+c_avg.jpg'))
