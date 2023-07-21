import torch
import torch.nn as nn

from Trainer_v03b3 import ImageEncoderM1, ImageDecoderM1
from TrainerTS import timer, MyDataset, split_loader, MyArgs, bn, Interpolate
from TrainerVTS import TrainerVTS


class AuxEncoder(nn.Module):
    def __init__(self):
        super(AuxEncoder, self).__init__()

