import torch
import torch.nn as nn
from torchinfo import summary


def reparameterize(mu, logvar):
    eps = torch.randn_like(mu)
    return mu + eps * torch.exp(logvar / 2)


class CsiEncoder(nn.Module):
    def __init__(self, latent_dim=16, feature_length=512):
        super(CsiEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.feature_length = 512

        self.cnn = nn.Sequential(
            # 2 * 90 * 100
            nn.Conv2d(2, 16, kernel_size=3, stride=(3, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            # 16 * 30 * 98
            nn.Conv2d(16, 64, kernel_size=3, stride=(2, 2), padding=0),
            nn.LeakyReLU(inplace=True),
            # 64 * 14 * 48
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            # 128 * 12 * 46
            nn.Conv2d(128, 256, kernel_size=3, stride=(1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            # 256 * 10 * 44
            nn.Conv2d(256, self.feature_length, kernel_size=3, stride=(1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            # 512 * 8 * 42
        )

        self.gap = nn.AdaptiveAvgPool2d(output_size=(self.feature_length, 42))

        self.lstm = nn.Sequential(
            nn.LSTM(self.feature_length, 2 * self.latent_dim, 2, batch_first=True, dropout=0.1),
        )

    def __str__(self):
        return 'CsiEnV0400'

    def forward(self, x):
        out = self.cnn(x)
        out = self.gap(out)
        out, (final_hidden_state, final_cell_state) = self.lstm.forward(
            out.view(-1, self.feature_length, 42).transpose(1, 2))

        if self.bottleneck == 'last':
            out = out[:, -1, :]

        mu, logvar = out.view(-1, 2 * self.latent_dim).chunk(2, dim=-1)
        z = reparameterize(mu, logvar)

        return out, z, mu, logvar


class CsiDecoder(nn.Module):
    def __init__(self, latent_dim=16):
        super(CsiDecoder, self).__init__()

        self.latent_dim = latent_dim

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 18000),
        )

    def __str__(self):
        return 'CsiDeV0400'

    def forward(self, x):
        out = self.fclayers(x)
        return out.view(-1, 2, 90, 100)
