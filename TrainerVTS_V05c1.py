import torch
import torch.nn as nn
from torchvision.ops import generalized_box_iou_loss
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from Loss import MyLoss, MyLoss_S


class TrainerVTS_V05c1:
    def __init__(self, img_encoder, img_decoder, csi_encoder,
                 lr, epochs, cuda,
                 train_loader, valid_loader, test_loader,
                 ):

        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda:" + str(cuda) if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam

        self.models = {'imgen': img_encoder.to(self.device),
                       'imgde': img_decoder.to(self.device),
                       'csien': csi_encoder.to(self.device)
                       }

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.alpha = 0.8
        self.beta = 1.2

        self.recon_lossfun = nn.MSELoss(reduction='sum')

        self.temp_loss = {}
        self.loss = {'t': MyLoss(loss_terms=['LOSS', 'KL_I', 'RECON_I', 'KL_B', 'RECON_B', 'BBX'],
                                       pred_terms=['GT', 'PRED', 'GT_BBX', 'PRED_BBX', 'IND']),
                     's': MyLoss_S(loss_terms=['LOSS', 'MU_I', 'LOGVAR_I', 'MU_B', 'LOGVAR_B', 'BBX', 'IMG'],
                                       pred_terms=['GT', 'T_PRED', 'S_PRED', 'T_LATENT_I', 'S_LATENT_I',
                                                   'T_LATENT_B', 'S_LATENT_B',
                                                   'GT_BBX', 'T_BBX', 'S_BBX', 'IND']),
                     }
        self.inds = None

    def current_title(self):
        """
        Shows current title
        :return: a string including current training epochs
        """
        return f"Te{self.loss['t'].epochs[-1]}_Se{self.loss['s'].epochs[-1]}"