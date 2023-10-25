import torch
import torch.nn as nn
from torchinfo import summary
from TrainerTS import bn, Interpolate

# -------------------------------------------------------------------------- #
# Models named with 'b' are AEs
# Models named with 'c' are VAEs
# Numbers after 'V' are generations
# Numbers after 'b' or 'c' are variations
# eg: ModelV03b1 means Gen3 AE Var1
# -------------------------------------------------------------------------- #


def reparameterize(mu, logvar):
    eps = torch.randn_like(mu)
    return mu + eps * torch.exp(logvar / 2)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

# -------------------------------------------------------------------------- #
# Model TS
# Model V03b1
# Added interpolating decoder
# Adaptive to MNIST

# ImageEncoder: in = 128 * 128, out = 1 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128
# CSIEncoder: in = 2 * 90 * 100, out = 1 * latent_dim
# -------------------------------------------------------------------------- #


class ImageEncoderV03b1(nn.Module):
    def __init__(self, bottleneck='fc', batchnorm=False, latent_dim=16, active_func=nn.Tanh()):
        super(ImageEncoderV03b1, self).__init__()

        self.bottleneck = bottleneck
        self.latent_dim = latent_dim
        self.active_func = active_func

        self.cnn = nn.Sequential(
            # In = 128 * 128 * 1
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 64 * 64 * 16

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            bn(32, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 32 * 32 * 32

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 16 * 16 * 64

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 8 * 8 * 128

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 4 * 4 * 256
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=(4, 4), stride=1, padding=0)
        )

        self.fclayers = nn.Sequential(
            nn.Linear(4 * 4 * 256, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.latent_dim),
            self.active_func
        )

    def __str__(self):
        return 'ImgEnV03b1' + self.bottleneck.capitalize()

    def forward(self, x):
        x = self.cnn(x)

        if self.bottleneck == 'fc':
            x = self.fclayers(x.view(-1, 4 * 4 * 256))
        elif self.bottleneck == 'gap':
            x = self.gap(x)
            x = nn.Sigmoid(x)

        return x.view(-1, self.latent_dim)


class ImageDecoderV03b1(nn.Module):
    def __init__(self, batchnorm=False, latent_dim=16, active_func=nn.Sigmoid()):
        super(ImageDecoderV03b1, self).__init__()

        self.latent_dim = latent_dim
        self.active_func = active_func

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256),
            nn.ReLU()
        )

        self.cnn = nn.Sequential(
            # In = 1 * 1 * 256
            nn.ConvTranspose2d(256, 128, kernel_size=4),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 4 * 4 * 128

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 8 * 8 * 64

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            bn(32, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 16 * 16 * 32

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 32 * 32 * 16

            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            bn(8, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 64 * 64 * 8

            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            bn(1, batchnorm),
            self.active_func,
            # Out = 128 * 128 * 1
        )

    def __str__(self):
        return 'ImgDeV03b1' + self.fc

    def forward(self, x):

        x = self.fclayers(x.view(-1, self.latent_dim))
        x = self.cnn(x.view(-1, 256, 1, 1))
        return x.view(-1, 1, 128, 128)


class ImageDecoderIntV03b1(ImageDecoderV03b1):
    def __init__(self, batchnorm=False, latent_dim=16, active_func=nn.Sigmoid()):
        super(ImageDecoderIntV03b1, self).__init__()

        self.latent_dim = latent_dim
        self.active_func = active_func

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU()
        )

        self.cnn = nn.Sequential(
            # In = 4 * 4 * 32
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            Interpolate(size=(16, 16)),
            # Out = 16 * 16 * 16

            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            bn(8, batchnorm),
            nn.LeakyReLU(inplace=True),
            Interpolate(size=(64, 64)),
            # Out = 64 * 64 * 8

            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            bn(4, batchnorm),
            nn.LeakyReLU(inplace=True),
            Interpolate(size=(128, 128)),
            # Out = 128 * 128 * 4

            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
            bn(1, batchnorm),
            self.active_func,
            # Out = 128 * 128 * 1
        )

    def __str__(self):
        return 'ImgDeIntV03b1' + self.fc

    def forward(self, x):

        x = self.fclayers(x.view(-1, self.latent_dim))
        x = self.cnn(x.view(-1, 32, 4, 4))
        return x.view(-1, 1, 128, 128)


class CsiEncoderV03b1(nn.Module):
    def __init__(self, bottleneck='last', batchnorm=False, latent_dim=16, feature_length=512):
        super(CsiEncoderV03b1, self).__init__()

        self.bottleneck = bottleneck
        self.latent_dim = latent_dim
        self.feature_length = feature_length

        self.cnn1 = nn.Sequential(
            # In = 90 * 100 * 1
            nn.Conv2d(1, 16, kernel_size=3, stride=(3, 1), padding=0),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 30 * 98 * 16
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 2), padding=0),
            bn(32, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 14 * 48 * 32
            nn.Conv2d(32, 64, kernel_size=3, stride=(1, 1), padding=0),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 12 * 46 * 64
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=0),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 10 * 44 * 128
            nn.Conv2d(128, 256, kernel_size=3, stride=(1, 1), padding=0),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 8 * 42 * 256
        )

        self.cnn2 = nn.Sequential(
            # In = 90 * 100 * 1
            nn.Conv2d(1, 16, kernel_size=3, stride=(3, 1), padding=0),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 30 * 98 * 16
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 2), padding=0),
            bn(32, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 14 * 48 * 32
            nn.Conv2d(32, 64, kernel_size=3, stride=(1, 1), padding=0),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 12 * 46 * 64
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=0),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 10 * 44 * 128
            nn.Conv2d(128, 256, kernel_size=3, stride=(1, 1), padding=0),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 8 * 42 * 256
        )

        self.gap = nn.Sequential(
            nn.AvgPool1d(kernel_size=8 * 42, stride=1, padding=0)
        )

        self.fclayers = nn.Sequential(
            nn.Linear(256 * 8 * 42, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256),
            nn.ReLU()
        )

        self.lstm = nn.Sequential(
            nn.LSTM(self.feature_length, self.latent_dim, 2, batch_first=True, dropout=0.1),
        )

    def __str__(self):
        return 'CsiEnV03b1' + self.bottleneck.capitalize()

    def forward(self, x):
        x = torch.chunk(x.view(-1, 2, 90, 100), 2, dim=1)
        x1 = self.cnn1(x[0])
        x2 = self.cnn2(x[1])
        out = torch.cat([x1, x2], dim=1)
        out, (final_hidden_state, final_cell_state) = self.lstm.forward(
            out.view(-1, self.feature_length, 8 * 42).transpose(1, 2))

        if self.bottleneck == 'last':
            out = out[:, -1, :]

        return out


# -------------------------------------------------------------------------- #
# Model v03b2
# Minor modifications to Model v03b1
# In number of channels

# ImageEncoder: in = 128 * 128, out = 1 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128
# -------------------------------------------------------------------------- #

class ImageEncoderV03b2(ImageEncoderV03b1):
    def __init__(self, batchnorm=False):
        super(ImageEncoderV03b2, self).__init__(batchnorm=batchnorm)

        self.cnn = nn.Sequential(
            # In = 128 * 128 * 1
            nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 64 * 64 * 128

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 32 * 32 * 128

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 16 * 16 * 128

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 8 * 8 * 256

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 4 * 4 * 256
        )

    def __str__(self):
        return 'ImgEnV03b2' + self.bottleneck.capitalize()


class ImageDecoderV03b2(ImageDecoderV03b1):
    def __init__(self, batchnorm=False):
        super(ImageDecoderV03b2, self).__init__(batchnorm=batchnorm)

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU()
        )

        self.cnn = nn.Sequential(
            # In = 4 * 4 * 128
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 8 * 8 * 128

            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 16 * 16 * 128

            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 32 * 32 * 128

            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 64 * 64 * 128

            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            bn(1, batchnorm),
            self.active_func,
            # Out = 128 * 128 * 1
        )

    def __str__(self):
        return 'ImgDeV03b2'

    def forward(self, x):
        x = self.fclayers(x)
        x = self.cnn(x.view(-1, 128, 4, 4))
        return x.view(-1, 1, 128, 128)

# -------------------------------------------------------------------------- #
# Model VTS
# Model v03c1
# VAE version; Adaptive to MNIST
# Added interpolating decoder

# ImageEncoder: in = 128 * 128, out = 2 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128 (=V03b1)
# ImageDecoderInterp: in = 1 * latent_dim, out = 128 * 128 (=V03b1)
# CSIEncoder: in = 2 * 90 * 100, out = 2 * latent_dim
# -------------------------------------------------------------------------- #


class ImageEncoderV03c1(ImageEncoderV03b1):
    def __init__(self, batchnorm=False):
        super(ImageEncoderV03c1, self).__init__(batchnorm=batchnorm)

        self.fclayers = nn.Sequential(
            nn.Linear(4 * 4 * 256, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2 * self.latent_dim),
            self.active_func
        )

    def __str__(self):
        return 'ImgEnV03c1'

    def forward(self, x):
        x = self.cnn(x)

        if self.bottleneck == 'fc':
            x = self.fclayers(x.view(-1, 4 * 4 * 256))
        elif self.bottleneck == 'gap':
            x = self.gap(x)
            x = nn.Sigmoid(x)

        mu, logvar = x.view(-1, 2 * self.latent_dim).chunk(2, dim=-1)
        z = reparameterize(mu, logvar)

        return x, z


class ImageDecoderV03c1(ImageDecoderV03b1):
    def __init__(self):
        super(ImageDecoderV03c1, self).__init__()

    def __str__(self):
        return 'ImgDeV03c1'


class ImageDecoderIntV03c1(ImageDecoderIntV03b1):
    def __init__(self, batchnorm=False):
        super(ImageDecoderIntV03c1, self).__init__(batchnorm=batchnorm)

    def __str__(self):
        return 'ImgDeIntV03c1'


class CsiEncoderV03c1(CsiEncoderV03b1):
    def __init__(self, batchnorm=False, feature_length=512):
        super(CsiEncoderV03c1, self).__init__(batchnorm=batchnorm, feature_length=feature_length)

        self.lstm = nn.Sequential(
            nn.LSTM(self.feature_length, 2 * self.latent_dim, 2, batch_first=True, dropout=0.1),
        )

    def __str__(self):
        return 'CsiEnV03c1' + self.bottleneck.capitalize()

    def forward(self, x):
        x = torch.chunk(x.view(-1, 2, 90, 100), 2, dim=1)
        x1 = self.cnn1(x[0])
        x2 = self.cnn2(x[1])
        out = torch.cat([x1, x2], dim=1)
        out, (final_hidden_state, final_cell_state) = self.lstm.forward(
            out.view(-1, self.feature_length, 8 * 42).transpose(1, 2))

        if self.bottleneck == 'last':
            out = out[:, -1, :]

        mu, logvar = out.view(-1, 2 * self.latent_dim).chunk(2, dim=-1)
        z = reparameterize(mu, logvar)

        return out, z

# -------------------------------------------------------------------------- #
# Model v03c2
# Minor modifications to Model v03c1
# In number of channels

# ImageEncoder: in = 128 * 128, out = 2 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128 (=V03b2)
# -------------------------------------------------------------------------- #


class ImageEncoderV03c2(ImageEncoderV03c1):
    def __init__(self, batchnorm=False):
        super(ImageEncoderV03c2, self).__init__()

        self.cnn = nn.Sequential(
            # In = 128 * 128 * 1
            nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 64 * 64 * 128

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 32 * 32 * 128

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 16 * 16 * 128

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 8 * 8 * 256

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 4 * 4 * 256
        )

    def __str__(self):
        return 'ImgEnV03c2' + self.bottleneck.capitalize()


class ImageDecoderV03c2(ImageDecoderV03b2):
    def __init__(self, batchnorm=False):
        super(ImageDecoderV03c2, self).__init__(batchnorm=batchnorm)

    def __str__(self):
        return 'ImgDeV03c2'


# -------------------------------------------------------------------------- #
# Model v03c3
# Minor modifications to Model v03c2
# In the use of Resblock
# CsiEncoder increased feature length

# ImageEncoder: in = 128 * 128, out = 2 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128
# ImageDecoderInterp: in = 1 * latent_dim, out = 128 * 128
# CSIEncoder: in = 2 * 90 * 100, out = 2 * latent_dim
# -------------------------------------------------------------------------- #


class ImageEncoderV03c3(ImageEncoderV03c2):
    def __init__(self, batchnorm=False):
        super(ImageEncoderV03c3, self).__init__(batchnorm=batchnorm)

        self.cnn = nn.Sequential(
            # In = 128 * 128 *
            nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 64 * 64 * 128

            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 32 * 32 * 64

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 16 * 16 * 128

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 8 * 8 * 256

            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 4 * 4 * 256
        )

    def __str__(self):
        return 'ImgEnV03c3' + self.bottleneck.capitalize()


class ImageDecoderV03c3(ImageDecoderV03c2):
    def __init__(self, batchnorm=False):
        super(ImageDecoderV03c3, self).__init__(batchnorm=batchnorm)

        self.cnn = nn.Sequential(
            # In = 4 * 4 * 128
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),

            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 8 * 8 * 128
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 16 * 16 * 128
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 32 * 32 * 128
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 64 * 64 * 128
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            bn(1, batchnorm),
            self.active_func,
            # Out = 128 * 128 * 1
        )


class ImageDecoderIntV03c3(ImageDecoderV03c2):
    def __init__(self, batchnorm=False):
        super(ImageDecoderIntV03c3, self).__init__(batchnorm=batchnorm)

        self.cnn = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            # 4x4x128
            Interpolate(size=(8, 8)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            # 6x6x64
            Interpolate(size=(12, 12)),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
            # 10x10x32
            Interpolate(size=(20, 20)),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0),
            # 18x18x16
            Interpolate(size=(36, 36)),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=0),
            # 34x34x8
            Interpolate(size=(68, 68)),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=0),
            # 66x66x4
            Interpolate(size=(132, 132)),
            nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=0),
            # 130x130x2
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=0),
            # 128x128x1
        )

    def __str__(self):
        return 'ImgDeIntV03c3'


class CsiEncoderV03c3(CsiEncoderV03c1):
    def __init__(self, batchnorm=False, feature_length=1024):
        super(CsiEncoderV03c3, self).__init__(batchnorm=batchnorm, feature_length=feature_length)

        self.cnn1 = nn.Sequential(
            # In = 90 * 100 * 1
            nn.Conv2d(1, 16, kernel_size=3, stride=(3, 1), padding=0),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 30 * 98 * 16
            nn.Conv2d(16, 64, kernel_size=3, stride=(2, 2), padding=0),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 14 * 48 * 64
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=0),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 12 * 46 * 128
            nn.Conv2d(128, 256, kernel_size=3, stride=(1, 1), padding=0),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 10 * 44 * 256
            nn.Conv2d(256, 512, kernel_size=3, stride=(1, 1), padding=0),
            bn(512, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 8 * 42 * 512
        )

        self.cnn2 = nn.Sequential(
            # In = 90 * 100 * 1
            nn.Conv2d(1, 16, kernel_size=3, stride=(3, 1), padding=0),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 30 * 98 * 16
            nn.Conv2d(16, 64, kernel_size=3, stride=(2, 2), padding=0),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 14 * 48 * 64
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=0),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 12 * 46 * 128
            nn.Conv2d(128, 256, kernel_size=3, stride=(1, 1), padding=0),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 10 * 44 * 256
            nn.Conv2d(256, 512, kernel_size=3, stride=(1, 1), padding=0),
            bn(512, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 8 * 42 * 512
        )

        self.fclayers = nn.Sequential(
            nn.Linear(512 * 8 * 42, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256),
            nn.ReLU()
        )

    def __str__(self):
        return 'CsiEnV03c3' + self.bottleneck.capitalize()


# -------------------------------------------------------------------------- #
# Model v04c1
# Implemented inductive biases

# ImageEncoder: in = 128 * 128, out = 2 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128 & 1 * inductive_dim
# CSIEncoder: in = 2 * 90 * 100, out = 2 * latent_dim
# -------------------------------------------------------------------------- #

class ImageEncoderV04c1(ImageEncoderV03c2):
    def __init__(self, batchnorm=False):
        super(ImageEncoderV04c1, self).__init__(batchnorm=batchnorm)

    def __str__(self):
        return 'ImgEnV04c1' + self.bottleneck.capitalize()


class ImageDecoderV04c1(ImageDecoderV03c2):
    def __init__(self, batchnorm=False):
        super(ImageDecoderV04c1, self).__init__(batchnorm=batchnorm)

    def __str__(self):
        return 'ImgDeV04c1'


if __name__ == "__main__":
    IMG = (1, 128, 128)
    CSI = (2, 90, 100)
    LAT = (1, 16)

    m = ImageEncoderV03c3()
    summary(m, input_size=IMG)
