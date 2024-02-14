import torch
import torch.nn as nn
from torchvision.ops import generalized_box_iou_loss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from Trainer import BasicTrainer
from Model import *
from Loss import MyLoss, MyLossBBX

version = 'V06C1'

##############################################################################
# -------------------------------------------------------------------------- #
# Version V06C1
# Teacher learns and estimates binary masks
# Student learns (6, 30, 30) CSIs

# ImageEncoder: in = 128 * 128, out = [latent_dim, latent_dim, latent_dim]
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128
# CSIEncoder: in = 6 * 30 * 30, out = [latent_dim, latent_dim, latent_dim]
# -------------------------------------------------------------------------- #
##############################################################################


class ImageEncoder(BasicImageEncoder):
    def __init__(self, *args, **kwargs):
        super(ImageEncoder, self).__init__(*args, **kwargs)

        channels = [1, 128, 128, 256, 256, 512]
        block = []
        for i in range(len(channels) - 1):
            block.extend([nn.Conv2d(channels[i], channels[i+1], 3, 2, 1),
                          batchnorm_layer(channels[i+1], self.batchnorm),
                          nn.LeakyReLU(inplace=True)])
        self.cnn = nn.Sequential(*block)

        # 1 * 128 * 128
        # 128 * 64 * 64
        # 128 * 32 * 32
        # 256 * 16 * 16
        # 256 * 8 * 8
        # 512 * 4 * 4

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
        return f"IMGEN{version}"

    def forward(self, x):
        out = self.cnn(x)
        mu = self.fc_mu(out.view(-1, 4 * 4 * 512))
        logvar = self.fc_logvar(out.view(-1, 4 * 4 * 512))
        z = reparameterize(mu, logvar)

        return z, mu, logvar


class ImageDecoder(BasicImageDecoder):
    def __init__(self, *args, **kwargs):
        super(ImageDecoder, self).__init__(*args, **kwargs)

        channels = [512, 256, 256, 128, 128, 1]
        block = []
        for i in range(len(channels) - 1):
            block.extend([nn.ConvTranspose2d(channels[i], channels[i+1], 4, 2, 1),
                          batchnorm_layer(channels[i+1], self.batchnorm),
                          nn.LeakyReLU(inplace=True)])
        # Replace the last LeakyReLU
        block.pop()
        self.cnn = nn.Sequential(*block, self.active_func)

        # 512 * 4 * 4
        # 256 * 8 * 8
        # 256 * 16 * 16
        # 128 * 32 * 32
        # 128 * 64 * 64
        # 1 * 128 * 128

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 8192),
            nn.ReLU()
        )

    def __str__(self):
        return f"IMGDE{version}"

    def forward(self, x):
        out = self.fclayers(x)
        out = self.cnn(out.view(-1, 512, 4, 4))
        return out.view(-1, 1, 128, 128)


class CSIEncoder(BasicCSIEncoder):
    def __init__(self, out_length, *args, **kwargs):
        super(CSIEncoder, self).__init__(*args, **kwargs)
        self.out_length = out_length
        self.cnn = nn.Sequential(
            nn.Conv2d(6, 128, 5, 1, 1),
            batchnorm_layer(128, self.batchnorm),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            batchnorm_layer(256, self.batchnorm),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1),
            batchnorm_layer(512, self.batchnorm),
            nn.LeakyReLU(inplace=True),
        )
        self.fclayers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2 * self.latent_dim)
            # nn.ReLU()
        )

    def __str__(self):
        return f"CSIEN{version}"

    def forward(self, x):
        out = self.cnn(x)
        out = self.fclayers(out.view(-1, 512 * 7 * 7))

        if self.out_length == 2 * self.latent_dim:
            mu_i, logvar_i = out.view(-1, 2 * self.latent_dim).chunk(2, dim=-1)
            z_i = reparameterize(mu_i, logvar_i)
            return z_i, mu_i, logvar_i
        else:
            bbx = out
            return bbx


class TeacherTrainer(BasicTrainer):
    def __init__(self,
                 beta=1.2,
                 recon_lossfunc=nn.BCELoss(reduction='sum'),
                 *args, **kwargs):
        super(TeacherTrainer, self).__init__(*args, **kwargs)

        self.modality = {'img'}

        self.beta = beta
        self.recon_lossfunc = recon_lossfunc

        self.loss_terms = {'LOSS', 'KL', 'RECON'}
        self.pred_terms = {'GT', 'PRED', 'LAT', 'IND'}
        self.loss = MyLoss(loss_terms=self.loss_terms,
                           pred_terms=self.pred_terms)

    def vae_loss(self, pred, gt, mu, logvar):
        recon_loss = self.recon_lossfunc(pred, gt) / pred.shape[0]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss * self.beta
        return loss, kl_loss, recon_loss

    def calculate_loss(self, data):

        z, mu, logvar = self.models['imgen'](data['img'])
        output = self.models['imgde'](z)
        loss, kl_loss, recon_loss = self.vae_loss(output, data['img'], mu, logvar)

        self.temp_loss = {'LOSS': loss,
                          'KL': kl_loss,
                          'RECON': recon_loss
                          }
        return {'GT': data['img'],
                'PRED': output,
                'LAT': torch.cat((mu, logvar), -1),
                'IND': data['ind']
                }

    def plot_test(self, select_ind=None, select_num=8, autosave=False, notion=''):
        title = {'PRED': "Teacher Test IMG Predicts",
                 'LAT': "Teacher Latents",
                 'LOSS': "Teacher Test Loss"}
        filename = {term: f"{notion}_{self.name}_{term}@{self.current_ep()}.jpg" for term in ('PRED', 'LAT', 'LOSS')}
        save_path = f'../saved/{notion}/'

        if select_ind:
            inds = select_ind
        else:
            if self.inds is not None:
                inds = self.inds
            else:
                inds = self.generate_indices(self.loss.loss['pred']['IND'], select_num)

        fig1 = self.loss.plot_predict(title['PRED'], inds, ('GT', 'PRED'))
        #fig2 = self.loss.plot_latent(title['LAT'], inds, ('LAT'))
        fig3 = self.loss.plot_test(title['LOSS'], inds)

        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig1.savefig(f"{save_path}{filename['PRED']}")
            #fig2.savefig(f"{save_path}{filename['LAT']}")
            fig3.savefig(f"{save_path}{filename['LOSS']}")


class StudentTrainer(BasicTrainer):
    def __init__(self,
                 alpha=0.8,
                 recon_lossfunc=nn.MSELoss(reduction='sum'),
                 *args, **kwargs):
        super(StudentTrainer, self).__init__(*args, **kwargs)

        self.modality = {'csi', 'img'}

        self.alpha = alpha
        self.recon_lossfunc = recon_lossfunc

        self.loss_terms = {'LOSS', 'MU', 'LOGVAR', 'IMG'}
        self.pred_terms = {'GT', 'T_PRED', 'S_PRED', 'T_LATENT', 'S_LATENT', 'IND'}
        self.loss = MyLoss(loss_terms=self.loss_terms,
                           pred_terms=self.pred_terms)

    def kd_loss(self, mu_s, logvar_s, mu_t, logvar_t):
        mu_loss = self.recon_lossfunc(mu_s, mu_t) / mu_s.shape[0]
        logvar_loss = self.recon_lossfunc(logvar_s, logvar_t) / logvar_s.shape[0]
        loss = self.alpha * mu_loss + (1 - self.alpha) * logvar_loss
        return loss, mu_loss, logvar_loss

    def calculate_loss(self, data):

        s_z, s_mu, s_logvar = self.models['csien'](data['csi'])

        with torch.no_grad():
            t_z, t_mu, t_logvar = self.models['imgen'](data['img'])
            s_output = self.models['imgde'](s_z)
            t_output = self.models['imgde'](t_z)
            image_loss = self.recon_lossfunc(s_output, data['img'])

        loss_i, mu_loss_i, logvar_loss_i = self.kd_loss(s_mu, s_logvar, t_mu, t_logvar)
        loss = loss_i

        self.temp_loss = {'LOSS': loss,
                          'MU': mu_loss_i,
                          'LOGVAR': logvar_loss_i,
                          'IMG': image_loss}
        return {'GT': data['img'],
                'T_LATENT': torch.cat((t_mu, t_logvar), -1),
                'S_LATENT': torch.cat((s_mu, s_logvar), -1),
                'T_PRED': t_output,
                'S_PRED': s_output,
                'IND': data['ind']}

    def plot_test(self, select_ind=None, select_num=8, autosave=False, notion=''):
        title = {'PRED': "Student Test IMG Predicts",
                 'LOSS': "Student Test Loss",
                 'LATENT': f"Student Test Latents for IMG"}
        filename = {term: f"{notion}_{self.name}_{term}@{self.current_ep()}.jpg" for term in ('PRED', 'LAT', 'LOSS')}

        save_path = f'../saved/{notion}/'

        if select_ind:
            inds = select_ind
        else:
            if self.inds is not None:
                inds = self.inds
            else:
                inds = self.generate_indices(self.loss.loss['pred']['IND'], select_num)

        fig1 = self.loss.plot_predict(title['PRED'], inds, ('GT', 'T_PRED', 'S_PRED'))
        fig3 = self.loss.plot_latent(title['LATENT'], inds, ('T_LATENT', 'S_LATENT'))
        fig4 = self.loss.plot_test(title['LOSS'], inds)

        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig1.savefig(f"{save_path}{filename['PRED']}")
            fig3.savefig(f"{save_path}{filename['LAT']}")
            fig4.savefig(f"{save_path}{filename['LOSS']}")


class StudentTrainerBBX(StudentTrainer):
    def __init__(self,
                 *args, **kwargs):
        super(StudentTrainerBBX, self).__init__(*args, **kwargs)

    @staticmethod
    def bbx_loss(bbx1, bbx2):
        # x, y, w, h to x1, y1, x2, y2
        # bbx1[..., -1] = bbx1[..., -1] + bbx1[..., -3]
        # bbx1[..., -2] = bbx1[..., -2] + bbx1[..., -4]
        # bbx2[..., -1] = bbx2[..., -1] + bbx2[..., -3]
        # bbx2[..., -2] = bbx2[..., -2] + bbx2[..., -4]
        return generalized_box_iou_loss(bbx1, bbx2, reduction='sum')

    def calculate_loss(self, data):

        s_bbx = self.models['csien'](data['csi'])
        bbx_loss = self.bbx_loss(s_bbx, data['bbx'])
        loss = bbx_loss

        self.temp_loss = {'LOSS': loss,
                          'BBX': bbx_loss}
        return {'GT': data['img'],
                'GT_BBX': data['bbx'],
                'S_BBX': s_bbx,
                'IND': data['ind']}

    def plot_test(self, select_ind=None, select_num=8, autosave=False, notion=''):
        title = {'BBX': "Student Test BBX Predicts"}
        filename = {'BBX': f"{notion}_{self.name}_BBX@{self.current_ep()}.jpg"}

        save_path = f'../saved/{notion}/'

        if select_ind:
            inds = select_ind
        else:
            if self.inds is not None:
                inds = self.inds
            else:
                inds = self.generate_indices(self.loss.loss['pred']['IND'], select_num)

        fig2 = self.loss['s'].plot_bbx(title['BBX'], inds)

        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig2.savefig(f"{save_path}{filename['BBX']}")


class TeacherTrainerMask(TeacherTrainer):
    def __init__(self,
                 *args, **kwargs):
        super(TeacherTrainerMask, self).__init__(*args, **kwargs)

    def calculate_loss(self, data):
        mask = torch.where(data['img'] > 0, 1., 0.)
        z, mu, logvar = self.models['imgen'](mask)
        output = self.models['imgde'](z)
        loss, kl_loss, recon_loss = self.vae_loss(output, mask, mu, logvar)

        self.temp_loss = {'LOSS': loss,
                          'KL': kl_loss,
                          'RECON': recon_loss
                          }
        return {'GT': mask,
                'PRED': output,
                'LAT': torch.cat((mu, logvar), -1),
                'IND': data['ind']
                }


class StudentTrainerMask(StudentTrainer):
    def __init__(self,
                 *args, **kwargs):
        super(StudentTrainerMask, self).__init__(*args, **kwargs)

    def calculate_loss(self, data):
        mask = torch.where(data['img'] > 0, 1., 0.)
        s_z, s_mu, s_logvar = self.models['csien'](data['csi'])

        with torch.no_grad():
            t_z, t_mu, t_logvar = self.models['imgen'](mask)
            s_output = self.models['imgde'](s_z)
            t_output = self.models['imgde'](t_z)
            image_loss = self.recon_lossfunc(s_output, mask)

        loss_i, mu_loss_i, logvar_loss_i = self.kd_loss(s_mu, s_logvar, t_mu, t_logvar)
        loss = loss_i

        self.temp_loss = {'LOSS': loss,
                          'MU': mu_loss_i,
                          'LOGVAR': logvar_loss_i,
                          'IMG': image_loss}
        return {'GT': mask,
                'T_LATENT': torch.cat((t_mu, t_logvar), -1),
                'S_LATENT': torch.cat((s_mu, s_logvar), -1),
                'T_PRED': t_output,
                'S_PRED': s_output,
                'IND': data['ind']}


if __name__ == '__main__':
    cc = CSIEncoder(out_length=32)
    summary(cc, input_size=CSI2)