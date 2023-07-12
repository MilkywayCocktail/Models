import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from torchvision import datasets

y = [1,2,3,4,5,6]
y2 = [4,5,7,6,7,1]
l = list(range(len(y)))

mnist = datasets.MNIST(root='../../dataset/MNIST/',
                       train=True,
                       download=True)

print(mnist[0])

import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class CVAE(nn.Module):
    def __init__(self, z_dim):
        super(CVAE, self).__init__()
        self.conv_enc1 = nn.Conv2d(1, 128, 3)
        self.conv_enc2 = nn.Conv2d(128, 128, 3)
        self.conv_enc3 = nn.Conv2d(128, 128, 3)
        self.flatten = nn.Flatten()
        hw = int((28 - 2 - 2 - 2))
        dim = int(128 * hw * hw)
        self.dense_encmean = nn.Linear(dim, z_dim)
        self.dense_encvar = nn.Linear(dim, z_dim)
        self.dense_dec1 = nn.Linear(z_dim, dim)
        self.unflatten = nn.Unflatten(1, torch.Size([128, hw, hw]))
        self.conv_dec1 = nn.ConvTranspose2d(128, 128, 3)
        self.conv_dec2 = nn.ConvTranspose2d(128, 128, 3)
        self.conv_dec3 = nn.ConvTranspose2d(128, 1, 3)
        self.ReLU = nn.LeakyReLU()
        self.out_activation = nn.Sigmoid()
        self.rec_loss = nn.BCELoss(reduction="sum")
        # self.rec_loss = nn.MSELoss(reduction="sum")
    def _encoder(self, input):
        x = self.ReLU(self.conv_enc1(input))
        x = self.ReLU(self.conv_enc2(x))
        x = self.ReLU(self.conv_enc3(x))
        x = self.flatten(x)
        mean = self.dense_encmean(x)
        logvar = self.dense_encvar(x)
        return mean, logvar
    def _sample_z(self, mean, logvar):
        epsilon = torch.randn(mean.shape).to(device)
        return mean + torch.exp(0.5 * logvar) * epsilon
    def _decoder(self, z):
        x = self.ReLU(self.dense_dec1(z))
        x = self.unflatten(x)
        x = self.ReLU(self.conv_dec1(x))
        x = self.ReLU(self.conv_dec2(x))
        x = self.out_activation(self.conv_dec3(x))
        return x
    def forward(self, x):
        mean, logvar = self._encoder(x)
        z = self._sample_z(mean, logvar)
        y = self._decoder(z)
        return y, z
    def loss(self, x):
        mean, logvar = self._encoder(x)
        KL = -0.5 * torch.mean(torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar)))
        z = self._sample_z(mean, logvar)
        y = self._decoder(z)
        reconstruction = self.rec_loss(self.flatten(y), self.flatten(x))
        lower_bound = reconstruction + KL
        return lower_bound, reconstruction, KL