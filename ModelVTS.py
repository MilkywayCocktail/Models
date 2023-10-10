import torch
import torch.nn as nn
from torchinfo import summary
from TrainerTS import bn, Interpolate


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


# ------------------------------------- #
# Model v03c1
# VAE version; Adaptive to MNIST
# Added interpolating decoder

# ImageEncoder: in = 128 * 128, out = 2 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128
# CSIEncoder: in = 2 * 90 * 100, out = 1 * 256 (Unused)


class ImageEncoderV03c1(nn.Module):
    def __init__(self, bottleneck='fc', batchnorm=False, latent_dim=8, active_func=nn.Tanh()):
        super(ImageEncoderV03c1, self).__init__()

        self.bottleneck = bottleneck
        self.latent_dim = latent_dim
        self.active_func = active_func

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 128 * 128 * 1
            # Out = 64 * 64 * 16

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            bn(32, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 64 * 64 * 16
            # Out = 32 * 32 * 32

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 32 * 32 * 32
            # Out = 16 * 16 * 64

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 16 * 16 * 64
            # Out = 8 * 8 * 128

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 8 * 8 * 128
            # Out = 4 * 4 * 256
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=(4, 4), stride=1, padding=0)
        )

        self.fclayers = nn.Sequential(
            nn.Linear(4 * 4 * 256, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2 * self.latent_dim),
            self.active_func
        )

    def __str__(self):
        return 'ImgEnV03c1' + self.bottleneck.capitalize()

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


class ImageDecoderV03c1(nn.Module):
    def __init__(self, batchnorm=False, latent_dim=8, active_func=nn.Sigmoid()):
        super(ImageDecoderV03c1, self).__init__()

        self.latent_dim = latent_dim
        self.active_func = active_func

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256),
            nn.ReLU()
        )

        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 1 * 1 * 256
            # Out = 4 * 4 * 128

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 4 * 4 * 128
            # Out = 8 * 8 * 64

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            bn(32, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 8 * 8 * 64
            # Out = 16 * 16 * 32

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 16 * 16 * 32
            # Out = 32 * 32 * 16

            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            bn(8, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 32 * 32 * 16
            # Out = 64 * 64 * 8

            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            bn(1, batchnorm),
            self.active_func,
            # In = 64 * 64 * 8
            # Out = 128 * 128 * 1
        )

    def __str__(self):
        return 'ImgDeV03c1'

    def forward(self, z):
        z = self.fclayers(z)
        z = self.cnn(z.view(-1, 256, 1, 1))

        return z.view(-1, 1, 128, 128)


class ImageDecoderIntV03c1(ImageDecoderV03c1):
    def __init__(self, batchnorm=False, latent_dim=8, active_func=nn.Sigmoid()):
        super(ImageDecoderIntV03c1, self).__init__(batchnorm, latent_dim, active_func)

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU()
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            Interpolate(size=(16, 16)),
            # In = 4 * 4 * 32
            # Out = 16 * 16 * 16

            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            bn(8, batchnorm),
            nn.LeakyReLU(inplace=True),
            Interpolate(size=(64, 64)),
            # In = 16 * 16 * 16
            # Out = 64 * 64 * 8

            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            bn(4, batchnorm),
            nn.LeakyReLU(inplace=True),
            Interpolate(size=(128, 128)),
            # In = 64 * 64 * 8
            # Out = 128 * 128 * 4

            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
            bn(1, batchnorm),
            self.active_func,
            # In = 128 * 128 * 4
            # Out = 128 * 128 * 1
        )

    def __str__(self):
        return 'ImgDeIntV03c1'

    def forward(self, z):
        z = self.fclayers(z)
        z = self.cnn(z.view(-1, 32, 4, 4))
        return z.view(-1, 1, 128, 128)


class CsiEncoderV03c1(nn.Module):
    def __init__(self, bottleneck='last', batchnorm=False, latent_dim=8, active_func=nn.Sigmoid(), feature_length=512):
        super(CsiEncoderV03c1, self).__init__()

        self.bottleneck = bottleneck
        self.latent_dim = latent_dim
        self.active_func = active_func
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

# Model v03c2
# Minor modifications to Model v03c1
# In number of channels

# ImageEncoder: in = 128 * 128, out = 1 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128
# CSIEncoder: in = 2 * 90 * 100, out = 1 * 256 (Unused)


class ImageEncoderV03c2(ImageEncoderV03c1):
    def __init__(self, bottleneck='fc', batchnorm=False, latent_dim=8, active_func=nn.Tanh()):
        super(ImageEncoderV03c2, self).__init__(bottleneck, batchnorm, latent_dim, active_func)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 128 * 128 * 1
            # Out = 64 * 64 * 128

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 64 * 64 * 128
            # Out = 32 * 32 * 128

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 32 * 32 * 128
            # Out = 16 * 16 * 128

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 16 * 16 * 128
            # Out = 8 * 8 * 256

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 8 * 8 * 256
            # Out = 4 * 4 * 256
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=(4, 4), stride=1, padding=0)
        )

        self.fclayers = nn.Sequential(
            nn.Linear(4 * 4 * 256, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2 * self.latent_dim),
            self.active_func
        )

    def __str__(self):
        return 'ImgEnV03c2' + self.bottleneck.capitalize()


class ImageDecoderV03c2(ImageDecoderV03c1):
    def __init__(self, batchnorm=False, latent_dim=8, active_func=nn.Sigmoid()):
        super(ImageDecoderV03c2, self).__init__(batchnorm, latent_dim, active_func)

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU()
        )

        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 4 * 4 * 128
            # Out = 8 * 8 * 128

            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 8 * 8 * 128
            # Out = 16 * 16 * 128

            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 16 * 16 * 128
            # Out = 32 * 32 * 128

            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 32 * 32 * 128
            # Out = 64 * 64 * 128

            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            bn(1, batchnorm),
            self.active_func,
            # In = 64 * 64 * 128
            # Out = 128 * 128 * 1
        )

    def __str__(self):
        return 'ImgDeV03c2'

    def forward(self, z):
        z = self.fclayers(z)
        z = self.cnn(z.view(-1, 128, 4, 4))

        return z.view(-1, 1, 128, 128)


class ImageEncoderV03c3(ImageEncoderV03c2):
    def __init__(self, bottleneck='fc', batchnorm=False, latent_dim=8, active_func=nn.Tanh()):
        super(ImageEncoderV03c3, self).__init__(bottleneck, batchnorm, latent_dim, active_func)

        self.cnn = nn.Sequential(
            # In = 128 * 128 * 1
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            bn(32, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # Out = 64 * 64 * 32

            # ResidualBlock(32, 32),
            # ResidualBlock(32, 32),
            # ResidualBlock(32, 32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # Out = 32 * 32 * 64

            # ResidualBlock(64, 64),
            # ResidualBlock(64, 64),
            # ResidualBlock(64, 64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # Out = 16 * 16 * 128

            # ResidualBlock(128, 128),
            # ResidualBlock(128, 128),
            # ResidualBlock(128, 128),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # Out = 8 * 8 * 256

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # Out = 4 * 4 * 256

            # ResidualBlock(256, 256),
            # ResidualBlock(256, 256)
        )

        self.fclayers = nn.Sequential(
            nn.Linear(4 * 4 * 256, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2 * self.latent_dim),
            self.active_func
        )

    def __str__(self):
        return 'ImgEnV03c3' + self.bottleneck.capitalize()

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


class ImageDecoderV03c3(ImageDecoderV03c2):
    def __init__(self, batchnorm=False, latent_dim=8, active_func=nn.Sigmoid()):
        super(ImageDecoderV03c3, self).__init__(batchnorm, latent_dim, active_func)

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU()
        )

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


class ImageDecoderIntV03c3(ImageDecoderV03c1):
    def __init__(self, batchnorm=False, latent_dim=8, active_func=nn.Sigmoid()):
        super(ImageDecoderIntV03c3, self).__init__(batchnorm, latent_dim, active_func)

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU()
        )

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

    def forward(self, z):
        z = self.fclayers(z)
        z = self.cnn(z.view(-1, 128, 4, 4))
        return z.view(-1, 1, 128, 128)


class CsiEncoderV03c3(CsiEncoderV03c1):
    def __init__(self, bottleneck='last', batchnorm=False, latent_dim=8, active_func=nn.Sigmoid(), feature_length=1024):
        super(CsiEncoderV03c3, self).__init__(bottleneck, batchnorm, latent_dim, active_func, feature_length)

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


if __name__ == "__main__":
    # m1 = ImageEncoder(batchnorm=False, latent_dim=256)
    # summary(m1, input_size=(1, 128, 128))
    # m2 = ImageDecoder(batchnorm=False)
    # summary(m2, input_size=(1, 16))
    # m3 = CsiEncoder(batchnorm=False)
    # summary(m3, input_size=(2, 90, 100))
    m4 = CsiEncoderV03c3(latent_dim=16)
    summary(m4, input_size=(2, 90, 100))
