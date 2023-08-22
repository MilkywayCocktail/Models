from Trainer_v03b2 import *


# ------------------------------------- #
# Model v03b3
# Minor modifications to Model v03b2
# In number of channels

# ImageEncoder: in = 128 * 128, out = 1 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128
# CSIEncoder: in = 2 * 90 * 100, out = 1 * 256 (Unused)

class ImageEncoderV03b3(ImageEncoder):
    def __init__(self, bottleneck='fc', batchnorm=False, latent_dim=8):
        super(ImageEncoderV03b3, self).__init__(bottleneck, batchnorm, latent_dim)

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 128 * 128 * 1
            # Out = 64 * 64 * 128
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 64 * 64 * 128
            # Out = 32 * 32 * 128
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 32 * 32 * 128
            # Out = 16 * 16 * 128
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 16 * 16 * 128
            # Out = 8 * 8 * 256
        )

        self.layer5 = nn.Sequential(
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
            # nn.Sigmoid()
            # nn.Tanh()
        )

    def __str__(self):
        return 'Model_v03b3_ImgEn_' + self.bottleneck.capitalize()


class ImageDecoderV03b3(ImageDecoder):
    def __init__(self, batchnorm=False, latent_dim=8, active_func=nn.Sigmoid()):
        super(ImageDecoderM1, self).__init__(batchnorm, latent_dim, active_func)

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 4 * 4 * 128
            # Out = 8 * 8 * 128
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 8 * 8 * 128
            # Out = 16 * 16 * 128
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 16 * 16 * 128
            # Out = 32 * 32 * 128
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 32 * 32 * 128
            # Out = 64 * 64 * 128
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            bn(1, batchnorm),
            self.active_func,
            # In = 64 * 64 * 128
            # Out = 128 * 128 * 1
        )

    def __str__(self):
        return 'Model_v03b3_ImgDe'

    def forward(self, z):
        z = self.fclayers(z)

        z = self.layer1(z.view(-1, 128, 4, 4))
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        z = self.layer5(z)

        return z.view(-1, 1, 128, 128)


if __name__ == "__main__":
    m1 = ImageDecoderM1(batchnorm=False, latent_dim=16)
    summary(m1, input_size=(1, 16))
