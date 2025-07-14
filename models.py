import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()

        # input is Z, going into a convolution
        self.convt1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8)

        # state size. (ngf*8) x 4 x 4
        self.convt2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 4)

        # state size. (ngf*4) x 8 x 8
        self.convt3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 2)

        # state size. (ngf*2) x 16 x 16
        self.convt4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)

        # state size. (ngf) x 32 x 32
        self.convt5 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)
        # state size. (nc) x 64 x 64

    def forward(self, x):
        x = self.bn1(self.convt1(x))
        x = F.relu(x, inplace=True)

        x = self.bn2(self.convt2(x))
        x = F.relu(x, inplace=True)

        x = self.bn3(self.convt3(x))
        x = F.relu(x, inplace=True)

        x = self.bn4(self.convt4(x))
        x = F.relu(x, inplace=True)

        x = self.convt5(x)
        x = torch.tanh_(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()

        # input is (nc) x 64 x 64
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)

        # state size. (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)

        # state size. (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 4)

        # state size. (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf * 8)

        # state size. (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2, inplace=True)

        x = self.bn2(self.conv2(x))
        x = F.leaky_relu(x, 0.2, inplace=True)

        x = self.bn3(self.conv3(x))
        x = F.leaky_relu(x, 0.2, inplace=True)

        x = self.bn4(self.conv4(x))
        x = F.leaky_relu(x, 0.2, inplace=True)

        x = self.conv5(x)
        x = torch.sigmoid(x)

        return x

class ConvFeatures(Discriminator):
    def __init__(self, nc, ndf):
        super(ConvFeatures, self).__init__(nc, ndf)

    def forward(self, x):
        x = self.conv1(x)
        c1 = x.clone().detach()
        x = F.leaky_relu(x, 0.2, inplace=True)

        x = self.conv2(x)
        c2 = x.clone().detach()
        x = F.leaky_relu(self.bn2(x), 0.2, inplace=True)

        x = self.conv3(x)
        c3 = x.clone().detach()
        x = F.leaky_relu(self.bn3(x), 0.2, inplace=True)

        c4 = self.conv4(x)

        return [c1, c2, c3, c4]

class Classifier(nn.Module):
    def __init__(self, input_size, n_classes):
        super(Classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.main(x)
