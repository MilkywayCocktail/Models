import torch
import torch.nn as nn
from torchinfo import summary
from TrainerTS import bn, Interpolate

# ------------------------------------- #
# Model TS
# Added interpolating decoder
# Adaptive to MNIST

# ImageEncoder: in = 128 * 128, out = 1 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128
# CSIEncoder: in = 2 * 90 * 100, out = 1 * 256


class ImageEncoderV03b1(nn.Module):
    def __init__(self, bottleneck='fc', batchnorm=False, latent_dim=8, active_func=nn.Tanh()):
        super(ImageEncoderV03b1, self).__init__()

        self.bottleneck = bottleneck
        self.latent_dim = latent_dim
        self.active_func = active_func

        self.cnn = nn.Sequential(
            # In = 128 * 128 * 1
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),

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
    def __init__(self, batchnorm=False, latent_dim=8, active_func=nn.Sigmoid()):
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
    def __init__(self, batchnorm=False, latent_dim=8, active_func=nn.Sigmoid()):
        super(ImageDecoderIntV03b1, self).__init__()

        self.latent_dim = latent_dim
        self.active_func = active_func

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256),
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
    def __init__(self, bottleneck='last', batchnorm=False, latent_dim=8):
        super(CsiEncoderV03b1, self).__init__()

        self.bottleneck = bottleneck
        self.latent_dim = latent_dim

        self.cnn = nn.Sequential(
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
            # nn.ReLU()
        )

        self.last_fc = nn.Linear(16, 16)

        self.lstm = nn.Sequential(
            nn.LSTM(512, self.latent_dim, 2, batch_first=True, dropout=0.1)
        )

    def __str__(self):
        return 'CsiEnV03b1' + self.bottleneck.capitalize()

    def forward(self, x):
        x = torch.chunk(x.view(-1, 2, 90, 100), 2, dim=1)
        x1 = self.cnn(x[0])
        x2 = self.cnn(x[1])

        out = torch.cat([x1, x2], dim=1)
        out, (final_hidden_state, final_cell_state) = self.lstm.forward(out.view(-1, 512, 8 * 42).transpose(1, 2))

        if self.bottleneck == 'full_fc':
            out = self.fclayers(out.view(-1, 256 * 8 * 42))

        elif self.bottleneck == 'full_gap':
            out = self.gap(out.transpose(1, 2))

        elif self.bottleneck == 'last':
            out = out[:, -1, :]

        elif self.bottleneck == 'last_fc':
            out = self.last_fc(out[:, -1, :])

        return out


# ------------------------------------- #
# Model v03b4
# Minor modifications to Model v03b1
# In number of channels

# ImageEncoder: in = 128 * 128, out = 1 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128
# CSIEncoder: in = 2 * 90 * 100, out = 1 * 256 (Unused)

class ImageEncoderV03b2(ImageEncoderV03b1):
    def __init__(self, bottleneck='fc', batchnorm=False, latent_dim=8, active_func=nn.Tanh()):
        super(ImageEncoderV03b2, self).__init__(bottleneck, batchnorm, latent_dim, active_func)

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
        return 'ImgEnV03b2' + self.bottleneck.capitalize()


class ImageDecoderV03b2(ImageDecoderV03b1):
    def __init__(self, batchnorm=False, latent_dim=8, active_func=nn.Sigmoid()):
        super(ImageDecoderV03b2, self).__init__(batchnorm, latent_dim, active_func)

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


if __name__ == "__main__":
    m1 = CsiEncoderV03b1(batchnorm=False, latent_dim=16)
    summary(m1, input_size=(2, 90, 100))
