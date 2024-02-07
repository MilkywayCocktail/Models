import torch
import torch.nn as nn
from torchinfo import summary

# -------------------------------------------------------------------------- #
# Models named with 'b' are AEs
# Models named with 'c' are VAEs
# Models named with 'i' have interpolation layers
# Numbers after 'V' are versions
# Numbers after 'b' or 'c' are variations
# eg: ModelV03b1 means Ver3 AE Var1
# -------------------------------------------------------------------------- #

def batchnorm_layer(channels, batchnorm=None):
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


class BasicCSIEncoder(nn.Module):
    name = 'csien'

    def __init__(self,  batchnorm=None, latent_dim=16, feature_length=512):
        super(BasicCSIEncoder, self).__init__()
        self.batchnowm = batchnorm
        self.latent_dim = latent_dim
        self.feature_length = feature_length

        self.cnn = None
        self.lstm = None

    def __str__(self):
        return 'CSIen'

    def forward(self, x):
        return x


class BasicImageEncoder(nn.Module):
    name = 'imgen'

    def __init__(self, batchnorm=None, latent_dim=16):
        super(BasicImageEncoder, self).__init__()
        self.batchnorm = batchnorm
        self.latent_dim = latent_dim

        self.cnn = None
        self.fclayers = None

    def __str__(self):
        return 'IMGen'

    def forward(self, x):
        return x


class BasicImageDecoder(nn.Module):
    name = 'imgde'

    def __init__(self, batchnorm=None, latent_dim=16, active_func=nn.Sigmoid()):
        super(BasicImageDecoder, self).__init__()
        self.batchnorm = batchnorm
        self.latent_dim = latent_dim
        self.active_func = active_func

        self.cnn = None
        self.fclayers = None

    def forward(self, x):
        return x