# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms

# Other libraries
import argparse
import random

# Add parent directory to path
import sys
sys.path.insert(0, '..')

# Local imports
from models import ConvFeatures, Classifier
from utils import Average


## Parse arguments
parser = argparse.ArgumentParser()
# Files and folders
parser.add_argument('--checkpoint', help='Path to checkpoint', required=True)
# Counts
parser.add_argument('--batch_size', default=128, type=int, help='Training batch size')
parser.add_argument('--num_epochs', default=25, type=int, help='Number of training epochs')
# Models parameters
parser.add_argument('--image_size', default=64, type=int, help='Spatial size of training images')
parser.add_argument('--nc', default=3, type=int, help='Number of channels in the training images')
parser.add_argument('--ndf', default=64, type=int, help='Discriminator feature maps size')
# Optimizers settings
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
# Others
parser.add_argument('--seed', default=None, type=int, help='Set seed for reproducibility, if not set it will be randomized')
args = parser.parse_args()


# If the seed has not been set, randomize it
if args.seed is None:
    args.seed = random.randint(1, 10000)
    print('Seed: {}'.format(args.seed))

# Set seed
random.seed(args.seed)
torch.manual_seed(args.seed)


# Create the datasets and dataloaders
transform = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
batch_size = 128
trainset = dset.CIFAR10(root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = dset.CIFAR10(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Load checkpoint
conv_features_net = ConvFeatures(args.nc, args.ndf).to(device)
checkpoint = torch.load(args.checkpoint)
conv_features_net.load_state_dict(checkpoint['discriminator_state_dict'])
conv_features_net.eval()

# Create fully connected model
classifier = Classifier(args.ndf * (120), len(classes)).to(device)  # 120 = 64 + 32 + 16 + 8
classifier.train()


# Initialize loss function
criterion = nn.CrossEntropyLoss()

# Create optimizer
optimizer = optim.Adam(classifier.parameters(), lr=args.lr)


# Training loop
for epoch in range(1, args.num_epochs + 1):

    running_loss = Average()

    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():

            # Get convolutional features
            features = conv_features_net(images)

            for j in range(len(features)):
                # Maxpool features
                features[j] = F.max_pool2d(features[j], 4)

                # Flatten features
                features[j] = torch.flatten(features[j], 1)

            # Concatenate layers of features
            features = torch.cat(features, dim=1)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = classifier(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss.update(loss.item())

    # Print statistics
    print('Epoch {}/{} - Loss: {}'.format(epoch, args.num_epochs, running_loss.mean()))


# Test the network
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        # Get convolutional features
        features = conv_features_net(images)

        for j in range(len(features)):
            # Maxpool features
            features[j] = F.max_pool2d(features[j], 4)

            # Flatten features
            features[j] = torch.flatten(features[j], 1)

        # Concatenate layers of features
        features = torch.cat(features, dim=1)

        # Calculate outputs
        outputs = classifier(features)

        # Predict classes
        _, predicted = torch.max(outputs.data, 1)

        # Update statistics
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: {}'.format(correct / total))
