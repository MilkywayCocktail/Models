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
        self.feature_length = feature_length

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

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 42))

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


class LatentEnTranslator(nn.Module):
    def __init__(self, latent_dim=16, repres_dim=128):
        super(LatentEnTranslator, self).__init__()

        self.latent_dim = latent_dim
        self.repres_dim = repres_dim

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.repres_dim),
            nn.ReLU(),
        )

    def __str__(self):
        return 'LatEnTrV0400'

    def forward(self, x):
        out = self.fclayers(x)

        return out


class LatentDeTranslator(nn.Module):
    def __init__(self, latent_dim=16, repres_dim=128):
        super(LatentDeTranslator, self).__init__()

        self.latent_dim = latent_dim
        self.repres_dim = repres_dim

        self.fclayers = nn.Sequential(
            nn.Linear(self.repres_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim),
            nn.ReLU(),
        )

    def __str__(self):
        return 'LatDeTrV0400'

    def forward(self, x):
        out = self.fclayers(x)

        return out


class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=16, active_func=nn.Tanh()):
        super(ImageEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.active_func = active_func

        self.cnn = nn.Sequential(
            # 1 * 128 * 128
            nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            # 128 * 64 * 64
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            # 128 * 32 * 32
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            # 128 * 16 * 16
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            # 256 * 8 * 8
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            # 256 * 4 * 4
        )

        self.fclayers = nn.Sequential(
            nn.Linear(4 * 4 * 256, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2 * self.latent_dim),
            self.active_func
        )

    def __str__(self):
        return 'ImgEnV0400'

    def forward(self, x):
        out = self.cnn(x)

        if self.bottleneck == 'fc':
            out = self.fclayers(out.view(-1, 4 * 4 * 256))
        elif self.bottleneck == 'gap':
            out = self.gap(out)
            out = nn.Sigmoid(out)

        mu, logvar = out.view(-1, 2 * self.latent_dim).chunk(2, dim=-1)
        z = reparameterize(mu, logvar)

        return out, z, mu, logvar


class ImageDecoder(nn.Module):
    def __init__(self, latent_dim=16):
        super(ImageDecoder, self).__init__()

        self.latent_dim = latent_dim

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU()
        )

        self.cnn = nn.Sequential(
            # 128 * 4 * 4
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            # 128 * 8 * 8
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            # 128 * 16 * 16
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            # 128 * 32 * 32
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            # 128 * 64 * 64
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            self.active_func,
            # 1 * 128 * 128
        )

    def __str__(self):
        return 'ImgDeV0400' + self.bottleneck.capitalize()

    def forward(self, x):

        out = self.fclayers(x.view(-1, self.latent_dim))
        out = self.cnn(out.view(-1, 256, 1, 1))
        return out.view(-1, 1, 128, 128)


if __name__ == "__main__":
    IMG = (1, 1, 128, 128)
    CSI = (2, 90, 100)
    LAT = (1, 16)

    m = CsiEncoder()
    summary(m, input_size=CSI)
