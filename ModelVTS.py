import torch
import torch.nn as nn
from torchinfo import summary

# -------------------------------------------------------------------------- #
# Models named with 'b' are AEs
# Models named with 'c' are VAEs
# Numbers after 'V' are generations
# Numbers after 'b' or 'c' are variations
# eg: ModelV03b1 means Gen3 AE Var1
# -------------------------------------------------------------------------- #


def bn(channels, batchnorm=False):
    """
    Definition of optional batchnorm layer.
    :param channels: input channels
    :param batchnorm: False or 'batch' or 'instance'
    :return: batchnorm layer or Identity layer (no batchnorm)
    """
    if not batchnorm:
        return nn.Identity(channels)
    elif batchnorm == 'batch':
        return nn.BatchNorm2d(channels)
    elif batchnorm == 'instance':
        return nn.InstanceNorm2d(channels)


class Interpolate(nn.Module):
    def __init__(self, size, mode='bilinear'):
        """
        Definition of interpolate layer.
        :param size: (height, width)
        :param mode: default is 'bilinear'
        """
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        out = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return out


def reparameterize(mu, logvar):
    eps = torch.randn_like(mu)
    return mu + eps * torch.exp(logvar / 2)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            bn(out_channels, batchnorm),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            bn(out_channels, batchnorm)
            )

        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        out = self.conv(x)
        out += x
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

        self.name = 'imgen'

        self.cnn = nn.Sequential(
            # 1 * 128 * 128
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 16 * 64 * 64
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            bn(32, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 32 * 32 * 32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 64 * 16 * 16 * 64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 8 * 8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 256 * 4 * 4
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
        out = self.cnn(x)

        if self.bottleneck == 'fc':
            out = self.fclayers(out.view(-1, 4 * 4 * 256))
        elif self.bottleneck == 'gap':
            out = self.gap(out)
            out = nn.Sigmoid(out)

        return out.view(-1, self.latent_dim)


class ImageDecoderV03b1(nn.Module):
    def __init__(self, batchnorm=False, latent_dim=16, active_func=nn.Sigmoid()):
        super(ImageDecoderV03b1, self).__init__()

        self.latent_dim = latent_dim
        self.active_func = active_func

        self.name = 'imgde'

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

        out = self.fclayers(x.view(-1, self.latent_dim))
        out = self.cnn(out.view(-1, 256, 1, 1))
        return out.view(-1, 1, 128, 128)


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

        out = self.fclayers(x.view(-1, self.latent_dim))
        out = self.cnn(out.view(-1, 32, 4, 4))
        return out.view(-1, 1, 128, 128)


class CsiEncoderV03b1(nn.Module):
    def __init__(self, bottleneck='last', batchnorm=False, latent_dim=16, feature_length=512):
        super(CsiEncoderV03b1, self).__init__()

        self.bottleneck = bottleneck
        self.latent_dim = latent_dim
        self.feature_length = feature_length

        self.name = 'csien'

        self.cnn1 = nn.Sequential(
            # 1 * 90 * 100
            nn.Conv2d(1, 16, kernel_size=3, stride=(3, 1), padding=0),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 16 * 30 * 98
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 2), padding=0),
            bn(32, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 32 * 14 * 48
            nn.Conv2d(32, 64, kernel_size=3, stride=(1, 1), padding=0),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 64 * 12 * 46
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=0),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 10 * 44
            nn.Conv2d(128, 256, kernel_size=3, stride=(1, 1), padding=0),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 256 * 8 * 42
        )

        self.cnn2 = nn.Sequential(
            # 1 * 90 * 100
            nn.Conv2d(1, 16, kernel_size=3, stride=(3, 1), padding=0),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 16 * 30 * 98
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 2), padding=0),
            bn(32, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 32 * 14 * 48
            nn.Conv2d(32, 64, kernel_size=3, stride=(1, 1), padding=0),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 64 * 12 * 46
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=0),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 10 * 44
            nn.Conv2d(128, 256, kernel_size=3, stride=(1, 1), padding=0),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 256 * 8 * 42
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
    def __init__(self, batchnorm=False, latent_dim=16):
        super(ImageEncoderV03b2, self).__init__(batchnorm=batchnorm, latent_dim=latent_dim)

        self.cnn = nn.Sequential(
            # 1 * 128 * 128
            nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 64 * 64
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 32 * 32
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 16 * 16
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 256 * 8 * 8
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 256 * 4 * 4
        )

    def __str__(self):
        return 'ImgEnV03b2' + self.bottleneck.capitalize()


class ImageDecoderV03b2(ImageDecoderV03b1):
    def __init__(self, batchnorm=False, active_func=nn.Sigmoid(), latent_dim=16):
        super(ImageDecoderV03b2, self).__init__(batchnorm=batchnorm, active_func=active_func, latent_dim=latent_dim)

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU()
        )

        self.cnn = nn.Sequential(
            # 128 * 4 * 4
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 8 * 8
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 16 * 16
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 32 * 32
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 64 * 64
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            bn(1, batchnorm),
            self.active_func,
            # 1 * 128 * 128
        )

    def __str__(self):
        return 'ImgDeV03b2'

    def forward(self, x):
        out = self.fclayers(x)
        out = self.cnn(out.view(-1, 128, 4, 4))
        return out.view(-1, 1, 128, 128)

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
    def __init__(self, batchnorm=False, latent_dim=16):
        super(ImageEncoderV03c1, self).__init__(batchnorm=batchnorm, latent_dim=latent_dim)

        self.fclayers = nn.Sequential(
            nn.Linear(4 * 4 * 256, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2 * self.latent_dim),
            self.active_func
        )

    def __str__(self):
        return 'ImgEnV03c1'

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


class ImageDecoderV03c1(ImageDecoderV03b1):
    def __init__(self, *args, **kwargs):
        super(ImageDecoderV03c1, self).__init__(*args, **kwargs)

    def __str__(self):
        return 'ImgDeV03c1'


class ImageDecoderIntV03c1(ImageDecoderIntV03b1):
    def __init__(self, *args, **kwargs):
        super(ImageDecoderIntV03c1, self).__init__(*args, **kwargs)

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

        return out, z, mu, logvar

# -------------------------------------------------------------------------- #
# Model v03c2
# Minor modifications to Model v03c1
# In number of channels

# ImageEncoder: in = 128 * 128, out = 2 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128 (=V03b2)
# -------------------------------------------------------------------------- #


class ImageEncoderV03c2(ImageEncoderV03c1):
    def __init__(self, batchnorm=False, latent_dim=16):
        super(ImageEncoderV03c2, self).__init__(batchnorm=batchnorm, latent_dim=latent_dim)

        self.cnn = nn.Sequential(
            # 1 * 128 * 128
            nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 64 * 64
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 32 * 32
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 16 * 16
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 256 * 8 * 8
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 256 * 4 * 4
        )

    def __str__(self):
        return 'ImgEnV03c2' + self.bottleneck.capitalize()


class ImageDecoderV03c2(ImageDecoderV03b2):
    def __init__(self, batchnorm=False, active_func=nn.Sigmoid(), latent_dim=16):
        super(ImageDecoderV03c2, self).__init__(batchnorm=batchnorm, active_func=active_func, latent_dim=latent_dim)

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )

        self.cnn = nn.Sequential(
            # 256 * 4 * 4
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 256 * 8 * 8
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 256 * 16 * 16
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 256 * 32 * 32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 64 * 64
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            bn(1, batchnorm),
            self.active_func,
            # 1 * 128 * 128
        )

    def __str__(self):
        return 'ImgDeV03c2'

    def forward(self, x):
        out = self.fclayers(x)
        out = self.cnn(out.view(-1, 256, 4, 4))
        return out.view(-1, 1, 128, 128)


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
    def __init__(self, batchnorm=False, latent_dim=16):
        super(ImageEncoderV03c3, self).__init__(batchnorm=batchnorm, latent_dim=latent_dim)

        self.cnn = nn.Sequential(
            # In = 128 * 128 *
            nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 64 * 64 * 128

            ResidualBlock(128, 128, batchnorm),
            ResidualBlock(128, 128, batchnorm),
            ResidualBlock(128, 128, batchnorm),

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

            ResidualBlock(256, 256, batchnorm),
            ResidualBlock(256, 256, batchnorm),
            ResidualBlock(256, 256, batchnorm),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # Out = 4 * 4 * 256
        )

    def __str__(self):
        return 'ImgEnV03c3' + self.bottleneck.capitalize()


class ImageDecoderV03c3(ImageDecoderV03c2):
    def __init__(self, batchnorm=False, latent_dim=16):
        super(ImageDecoderV03c3, self).__init__(batchnorm=batchnorm, latent_dim=latent_dim)

        self.cnn = nn.Sequential(
            # In = 4 * 4 * 128
            ResidualBlock(128, 128, batchnorm),
            ResidualBlock(128, 128, batchnorm),
            ResidualBlock(128, 128, batchnorm),

            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 8 * 8
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 16 * 16
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 32 * 32
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 64 * 64
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            bn(1, batchnorm),
            self.active_func,
            # 1 * 128 * 128
        )


class ImageDecoderIntV03c3(ImageDecoderV03c2):
    def __init__(self, batchnorm=False, latent_dim=16):
        super(ImageDecoderIntV03c3, self).__init__(batchnorm=batchnorm, latent_dim=latent_dim)

        self.cnn = nn.Sequential(
            ResidualBlock(128, 128, batchnorm),
            ResidualBlock(128, 128, batchnorm),
            ResidualBlock(128, 128, batchnorm),
            # 128 * 4 * 4
            Interpolate(size=(8, 8)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            # 64 * 6 * 6
            Interpolate(size=(12, 12)),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
            # 32 * 10 * 10
            Interpolate(size=(20, 20)),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0),
            # 16 * 18 * 18
            Interpolate(size=(36, 36)),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=0),
            # 8 * 34 * 34
            Interpolate(size=(68, 68)),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=0),
            # 4 * 66 * 66
            Interpolate(size=(132, 132)),
            nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=0),
            # 2 * 130 * 130
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=0),
            # 1 * 128 * 128
        )

    def __str__(self):
        return 'ImgDeIntV03c3'


class CsiEncoderV03c3(CsiEncoderV03c1):
    def __init__(self, batchnorm=False, feature_length=1024):
        super(CsiEncoderV03c3, self).__init__(batchnorm=batchnorm, feature_length=feature_length)

        self.cnn1 = nn.Sequential(
            # 1 * 90 * 100
            nn.Conv2d(1, 16, kernel_size=3, stride=(3, 1), padding=0),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 16 * 30 * 98
            nn.Conv2d(16, 64, kernel_size=3, stride=(2, 2), padding=0),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 64 * 14 * 48
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=0),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 12 * 46
            nn.Conv2d(128, 256, kernel_size=3, stride=(1, 1), padding=0),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 256 * 10 * 44
            nn.Conv2d(256, 512, kernel_size=3, stride=(1, 1), padding=0),
            bn(512, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 512 * 8 * 42
        )

        self.cnn2 = nn.Sequential(
            # 1 * 90 * 100
            nn.Conv2d(1, 16, kernel_size=3, stride=(3, 1), padding=0),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 16 * 30 * 98
            nn.Conv2d(16, 64, kernel_size=3, stride=(2, 2), padding=0),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 64 * 14 * 48
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=0),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 12 * 46
            nn.Conv2d(128, 256, kernel_size=3, stride=(1, 1), padding=0),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 256 * 10 * 44
            nn.Conv2d(256, 512, kernel_size=3, stride=(1, 1), padding=0),
            bn(512, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 512 * 8 * 42
        )

    def __str__(self):
        return 'CsiEnV03c3' + self.bottleneck.capitalize()

# -------------------------------------------------------------------------- #
# Model v03c4
# Modified CSI Encoder (2-channeled)

# ImageEncoder: in = 128 * 128, out = 2 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128 & 1 * inductive_dim
# CSIEncoder: in = 2 * 90 * 100, out = 2 * latent_dim
# -------------------------------------------------------------------------- #


class CsiEncoderV03c4(CsiEncoderV03c1):
    def __init__(self, batchnorm=False, feature_length=512):
        super(CsiEncoderV03c4, self).__init__(batchnorm=batchnorm, feature_length=feature_length)

        self.cnn = nn.Sequential(
            # 2 * 90 * 100
            nn.Conv2d(2, 32, kernel_size=3, stride=(3, 1), padding=0),
            bn(32, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 32 * 30 * 98
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=0),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 64 * 14 * 48
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=0),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 12 * 46
            nn.Conv2d(128, 256, kernel_size=3, stride=(1, 1), padding=0),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 256 * 10 * 44
            nn.Conv2d(256, self.feature_length, kernel_size=3, stride=(1, 1), padding=0),
            bn(self.feature_length, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 512 * 8 * 42
        )

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 42))
        # self.gap2 = nn.AvgPool1d(kernel_size=8, stride=1, padding=0)

        self.lstm = nn.Sequential(
            nn.LSTM(self.feature_length, 2 * self.latent_dim, 2, batch_first=True, dropout=0.1),
        )

        # self.fclayers = nn.Sequential(
        #    nn.Linear(self.feature_length, 128),
        #    nn.ReLU(),
        #    nn.Linear(128, 32),
        #    nn.Tanh()
        #)

    def __str__(self):
        return 'CsiEnV03c4' + self.bottleneck.capitalize()

    def forward(self, x):
        out = self.cnn(x)
        out = self.gap(out)
        # out = self.fclayers(out.view(-1, 512))
        out, (final_hidden_state, final_cell_state) = self.lstm.forward(
            out.view(-1, self.feature_length, 42).transpose(1, 2))

        if self.bottleneck == 'last':
            out = out[:, -1, :]

        mu, logvar = out.view(-1, 2 * self.latent_dim).chunk(2, dim=-1)
        z = reparameterize(mu, logvar)

        return out, z, mu, logvar


# -------------------------------------------------------------------------- #
# Model v03d1
# All dense layers

# ImageEncoder: in = 128 * 128, out = 2 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128 & 1 * inductive_dim
# CSIEncoder: in = 2 * 90 * 100, out = 2 * latent_dim
# -------------------------------------------------------------------------- #


class ImageEncoderV03d1(ImageEncoderV03c1):
    def __init__(self):
        super(ImageEncoderV03d1, self).__init__()

        self.fclayers = nn.Sequential(
            nn.Linear(128 * 128, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * self.latent_dim),
            nn.ReLU()
        )

    def __str__(self):
        return 'ImgEnV03d1' + self.bottleneck.capitalize()

    def forward(self, x):
        out = self.fclayers(x.view(-1, 128 * 128))

        mu, logvar = out.view(-1, 2 * self.latent_dim).chunk(2, dim=-1)
        z = reparameterize(mu, logvar)

        return out, z, mu, logvar


class ImageDecoderV03d1(ImageDecoderV03c1):
    def __init__(self):
        super(ImageDecoderV03d1, self).__init__()

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 8192),
            nn.ReLU(),
            nn.Linear(8192, 16384),
            nn.ReLU()
        )

    def __str__(self):
        return 'ImgDeV03d1'

    def forward(self, x):
        out = self.fclayers(x)
        return out.view(-1, 1, 128, 128)


# -------------------------------------------------------------------------- #
# Model v04c1
# Implemented inductive biases

# ImageEncoder: in = 128 * 128, out = 2 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128 & 1 * inductive_dim
# CSIEncoder: in = 2 * 90 * 100, out = 2 * latent_dim
# -------------------------------------------------------------------------- #


class ImageEncoderV04c1(ImageEncoderV03c2):
    def __init__(self, *args, **kwargs):
        super(ImageEncoderV04c1, self).__init__(*args, **kwargs)

    def __str__(self):
        return 'ImgEnV04c1' + self.bottleneck.capitalize()


class ImageDecoderV04c1(ImageDecoderV03c2):
    def __init__(self, batchnorm=False, inductive_length=25):
        super(ImageDecoderV04c1, self).__init__(batchnorm=batchnorm)

        self.inductive_length = inductive_length
        self.fc2 = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.inductive_length),
            nn.ReLU()
        )

    def __str__(self):
        return 'ImgDeV04c1'

    def forward(self, x):
        ib = self.fc2(x)
        out = self.fclayers(x.view(-1, self.latent_dim))
        out = self.cnn(out.view(-1, 256, 1, 1))
        return out.view(-1, 1, 128, 128), ib.view(-1, self.inductive_length)

# -------------------------------------------------------------------------- #
# Model v04c2
# Estimate cropped subject and bounding box

# ImageEncoder: in = 128 * 128, out = 2 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128
# CSIEncoder: in = 2 * 90 * 100, out = 2 * latent_dim
# MaskEncoder: in = img_width * 128, out = 128
# MaskDecoder: in = 128, out = 4
# -------------------------------------------------------------------------- #


class ImageEncoderV04c2(ImageEncoderV03c2):
    def __init__(self, *args, **kwargs):
        super(ImageEncoderV04c2, self).__init__(*args, **kwargs)

    def __str__(self):
        return 'ImgEnV04c2' + self.bottleneck.capitalize()


class ImageDecoderV04c2(ImageDecoderV03c2):
    def __init__(self, batchnorm=False, active_func=nn.Sigmoid(), latent_dim=16):
        super(ImageDecoderV04c2, self).__init__(batchnorm=batchnorm, active_func=active_func, latent_dim=latent_dim)

        self.cnn = nn.Sequential(
            # 256 * 4 * 4
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 256 * 8 * 8
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 256 * 16 * 16
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 256 * 32 * 32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 64 * 64
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            #nn.LeakyReLU(inplace=True),
            # 128 * 128 * 128
            # nn.Conv2d(128, out_channels=1, kernel_size=3, stride=1, padding=1),
            self.active_func,
            # 1 * 128 * 128
        )

    def __str__(self):
        return 'ImgDeV04c2'


class MaskEncoderV04c2(nn.Module):
    def __init__(self, latent_dim=16, image_width=128):
        super(MaskEncoderV04c2, self).__init__()
        self.latent_dim = latent_dim
        self.image_width = image_width

        self.name = 'msken'

        self.lstm = nn.Sequential(
            nn.LSTM(128, 2 * self.latent_dim, 2, batch_first=True, dropout=0.1),
        )

    def __str__(self):
        return 'MskEnV04c2' + self.bottleneck.capitalize()

    def forward(self, x):
        out, (final_hidden_state, final_cell_state) = self.lstm(x.view(-1, 128, self.image_width).transpose(1, 2))
        out = out[:, -1, :]
        mu, logvar = out.view(-1, 2 * self.latent_dim).chunk(2, dim=-1)
        z = reparameterize(mu, logvar)

        return out, z, mu, logvar


class MaskDecoderV04c2(nn.Module):
    def __init__(self, latent_dim=16):
        super(MaskDecoderV04c2, self).__init__()
        self.latent_dim = latent_dim

        self.name = 'mskde'

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )

    def __str__(self):
        return 'MskDeV04c2'

    def forward(self, x):
        out = self.fc(x.view(-1, self.latent_dim))
        return out


class CsiEncoderV04c2(CsiEncoderV03c4):
    def __init__(self, batchnorm=False, feature_length=512, middle_length=128):
        super(CsiEncoderV04c2, self).__init__(batchnorm=batchnorm, feature_length=feature_length)

        self.middle_length = middle_length
        self.lstm = nn.Sequential(
            nn.LSTM(self.feature_length, 128, 2, batch_first=True, dropout=0.1),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.middle_length, self.latent_dim),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.middle_length, self.latent_dim),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(self.middle_length, self.latent_dim),
        )

        self.fc4 = nn.Sequential(
            nn.Linear(self.middle_length, self.latent_dim),
        )

    def __str__(self):
        return 'CsiEnV04c2' + self.bottleneck.capitalize()

    def forward(self, x):
        out = self.cnn(x)
        out = self.gap(out)
        out, (final_hidden_state, final_cell_state) = self.lstm.forward(
            out.view(-1, self.feature_length, 42).transpose(1, 2))

        if self.bottleneck == 'last':
            out = out[:, -1, :]

        # mu = out[:, :self.latent_dim]
        # logvar = out[:, self.latent_dim:2 * self.latent_dim]
        # bbx = out[:, 2 * self.latent_dim:]

        mu_i = self.fc1(out)
        logvar_i = self.fc2(out)
        mu_b = self.fc3(out)
        logvar_b = self.fc4(out)
        z_i = reparameterize(mu_i, logvar_i)
        z_b = reparameterize(mu_b, logvar_b)
        latent_i = torch.cat((mu_i, logvar_i), -1)
        latent_b = torch.cat((mu_b, logvar_b), -1)

        return z_i, latent_i, mu_i, logvar_i, z_b, latent_b, mu_b, logvar_b


# -------------------------------------------------------------------------- #
# Model v05c1
# Added channel attention
# CSIEncoder estimates BBX

# ImageEncoder: in = 128 * 128, out = 2 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128
# CSIEncoder: in = 2 * 90 * 100, out = 128
# -------------------------------------------------------------------------- #

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        proj_query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)

        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=2)

        proj_value = self.value_conv(x).view(batch_size, -1, height * width)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        out = self.gamma * out + x
        return out


class ImageEncoderV05c1(nn.Module):
    def __init__(self, batchnorm=False, latent_dim=16):
        super(ImageEncoderV05c1, self).__init__()
        self.latent_dim = latent_dim
        self.active_func = None

        self.name = 'imgen'

        self.cnn = nn.Sequential(
            # 1 * 128 * 128
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            bn(32, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 32 * 64 * 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 64 * 32 * 32
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 128 * 16 * 16
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 256 * 8 * 8
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            bn(512, batchnorm),
            nn.LeakyReLU(inplace=True),
            # 512 * 4 * 4
        )

        self.attn = SelfAttention(512)

        self.fc_mu = nn.Sequential(
            nn.Linear(4 * 4 * 512, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.latent_dim),
            # self.active_func
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(4 * 4 * 512, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.latent_dim),
            # self.active_func
        )

    def __str__(self):
        return 'ImgEnV05c1'

    def forward(self, x):
        out = self.cnn(x)
        out = self.attn(out)
        mu = self.fc_mu(out.view(-1, 4 * 4 * 512))
        logvar = self.fc_logvar(out.view(-1, 4 * 4 * 512))
        z = reparameterize(mu, logvar)

        return z, mu, logvar


class CsiEncoderV05c1(CsiEncoderV03c4):
    def __init__(self, batchnorm=False, feature_length=512, middle_length=128):
        super(CsiEncoderV05c1, self).__init__(batchnorm=batchnorm, feature_length=feature_length)

        self.middle_length = middle_length
        self.attn = SelfAttention(512)
        self.lstm = nn.Sequential(
            nn.LSTM(self.feature_length * 8, 128, 2, batch_first=True, dropout=0.1),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.middle_length, self.latent_dim),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.middle_length, self.latent_dim),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(self.middle_length, 4),
        )

    def __str__(self):
        return 'CsiEnV05c1'

    def forward(self, x):
        out = self.cnn(x)
        out = self.attn(out)
        # out = self.gap(out)
        out, (final_hidden_state, final_cell_state) = self.lstm.forward(
            out.view(-1, self.feature_length * 8, 42).transpose(1, 2))

        out = out[:, -1, :]

        mu_i = self.fc1(out)
        logvar_i = self.fc2(out)
        bbx = self.fc3(out)

        z_i = reparameterize(mu_i, logvar_i)

        return z_i, mu_i, logvar_i, bbx


# -------------------------------------------------------------------------- #
# Model v05c2
# Modify CSIEncoder

# ImageEncoder: in = 128 * 128, out = 2 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128
# CSIEncoder: in = 2 * 90 * 100, out = configurable
# -------------------------------------------------------------------------- #


class CsiEncoderV05c2(CsiEncoderV03c4):
    def __init__(self, batchnorm=False, feature_length=512, middle_length=128, out_length=0):
        super(CsiEncoderV05c2, self).__init__(batchnorm=batchnorm, feature_length=feature_length)

        self.middle_length = middle_length
        self.attn = SelfAttention(512)
        self.out_length = 2 * self.latent_dim if out_length == 0 else out_length

        self.lstm = nn.Sequential(
            nn.LSTM(self.feature_length * 8, self.out_length, 2, batch_first=True, dropout=0.1),
        )

    def __str__(self):
        return 'CsiEnV05c2'

    def forward(self, x):
        out = self.cnn(x)
        out = self.attn(out)
        # out = self.gap(out)
        out, (final_hidden_state, final_cell_state) = self.lstm.forward(
            out.view(-1, self.feature_length * 8, 42).transpose(1, 2))

        out = out[:, -1, :]

        if self.out_length == 2 * self.latent_dim:
            mu_i, logvar_i = out.view(-1, 2 * self.latent_dim).chunk(2, dim=-1)
            bbx = 0
            z_i = reparameterize(mu_i, logvar_i)
        else:
            mu_i, logvar_i, z_i = 0, 0, 0
            bbx = out

        return z_i, mu_i, logvar_i, bbx


# -------------------------------------------------------------------------- #
# Model v05c3
# Added AoA/ToF input

# ImageEncoder: in = 128 * 128, out = 2 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128
# CSIEncoder: in = 2 * 90 * 100, out = configurable
# PDEncoder: in = 2, out = 1 * latent
# -------------------------------------------------------------------------- #

class PDEncoderV05c3(nn.Module):
    def __init__(self, latent_dim=16, out_length=32):
        super(PDEncoderV05c3, self).__init__()
        self.latent_dim = latent_dim
        self.out_length = out_length

        self.name = 'pden'

        self.fc = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_length)
        )

    def __str__(self):
        return 'PDEnV05c3'

    def forward(self, x):
        out = self.fc(x)
        if self.out_length == 2 * self.latent_dim:
            mu_i, logvar_i = out.view(-1, 2 * self.latent_dim).chunk(2, dim=-1)
            bbx = 0
            z_i = reparameterize(mu_i, logvar_i)
        else:
            mu_i, logvar_i, z_i = 0, 0, 0
            bbx = out

        return z_i, mu_i, logvar_i, bbx


# -------------------------------------------------------------------------- #
# Model v05c4
# Aggregated AoA/ToF input and CSI input

# ImageEncoder: in = 128 * 128, out = 2 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128
# PDEncoder: in = 2, out = 1 * latent
# -------------------------------------------------------------------------- #

class CsiEncoderV05c4(CsiEncoderV03c4):
    def __init__(self, batchnorm=False, feature_length=512, middle_length=128, out_length=0):
        super(CsiEncoderV05c4, self).__init__(batchnorm=batchnorm, feature_length=feature_length)

        self.middle_length = middle_length
        self.attn = SelfAttention(512)
        self.out_length = 2 * self.latent_dim if out_length == 0 else out_length

        self.lstm = nn.Sequential(
            nn.LSTM(self.feature_length * 8, self.out_length, 2, batch_first=True, dropout=0.1),
        )

        self.fc = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_length)
        )

        self.out = nn.Sequential(
            nn.Linear(2 * self.out_length, self.out_length)
        )

    def __str__(self):
        return 'CsiEnV05c4'

    def forward(self, csi, pd):
        pd_mid = self.fc(pd)
        csi_mid = self.cnn(csi)
        csi_mid = self.attn(csi_mid)
        # out = self.gap(out)
        csi_mid, (final_hidden_state, final_cell_state) = self.lstm.forward(
            csi_mid.view(-1, self.feature_length * 8, 42).transpose(1, 2))

        csi_mid = csi_mid[:, -1, :]

        out = torch.cat((csi_mid, pd_mid), -1)
        out = self.out(out)

        if self.out_length == 2 * self.latent_dim:
            mu_i, logvar_i = out.view(-1, 2 * self.latent_dim).chunk(2, dim=-1)
            bbx = 0
            z_i = reparameterize(mu_i, logvar_i)
        else:
            mu_i, logvar_i, z_i = 0, 0, 0
            bbx = out

        return z_i, mu_i, logvar_i, bbx


if __name__ == "__main__":
    IMG = (1, 1, 128, 128)
    CSI = (1, 2, 90, 100)
    LAT = (1, 16)
    RIMG = (1, 1, 128, 226)
    PD = (1, 2)

    m = CsiEncoderV05c4()
    summary(m, input_size=[CSI, PD])
