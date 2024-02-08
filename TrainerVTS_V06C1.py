import torch
import torch.nn as nn
from torchvision.ops import generalized_box_iou_loss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.ops import generalized_box_iou_loss
import os
from Trainer import BasicTrainer
from Loss import MyLoss, MyLossBBX
from misc import timer


class TeacherTrainer(BasicTrainer):
    def __init__(self,
                 beta=1.2,
                 recon_lossfunc=nn.MSELoss(reduction='sum'),
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
                'IND': data['i']
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
                if select_num >= len(self.inds):
                    inds = self.inds
                else:
                    inds = self.generate_indices(self.loss.loss['pred']['IND'], select_num)

        fig1 = self.loss.plot_predict(title['PRED'], inds, ('GT', 'PRED'))
        fig2 = self.loss.plot_latent(title['LAT'], inds)
        fig3 = self.loss.plot_test(title['LOSS'], inds)

        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig1.savefig(f"{save_path}{filename['PRED']}")
            fig2.savefig(f"{save_path}{filename['LAT']}")
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

        s_z, s_mu, s_logvar, s_bbx = self.models['csien'](data['csi'])

        with torch.no_grad():
            t_z, t_mu, t_logvar = self.models['imgen'](data['c_img'])
            s_output = self.models['imgde'](s_z)
            t_output = self.models['imgde'](t_z)
            image_loss = self.recon_lossfunc(s_output, data['c_img'])

        loss_i, mu_loss_i, logvar_loss_i = self.kd_loss(s_mu, s_logvar, t_mu, t_logvar)
        loss = loss_i

        self.temp_loss = {'LOSS': loss,
                          'MU': mu_loss_i,
                          'LOGVAR': logvar_loss_i,
                          'IMG': image_loss}
        return {'GT': data['c_img'],
                'T_LATENT': torch.cat((t_mu, t_logvar), -1),
                'S_LATENT': torch.cat((s_mu, s_logvar), -1),
                'T_PRED': t_output,
                'S_PRED': s_output,
                'IND': data['ind']}

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
                if select_num >= len(self.inds):
                    inds = self.inds
                else:
                    inds = self.generate_indices(self.loss.loss['pred']['IND'], select_num)

        fig1 = self.loss.plot_predict(title['PRED'], inds, ('GT', 'PRED'))
        fig2 = self.loss.plot_latent(title['LAT'], inds)
        fig3 = self.loss.plot_test(title['LOSS'], inds)

        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig1.savefig(f"{save_path}{filename['PRED']}")
            fig2.savefig(f"{save_path}{filename['LAT']}")
            fig3.savefig(f"{save_path}{filename['LOSS']}")
