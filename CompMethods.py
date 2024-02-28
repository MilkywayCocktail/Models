import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from Trainer import BasicTrainer
from Loss import MyLoss
from Model import ResidualBlock, Interpolate, reparameterize

##############################################################################
# -------------------------------------------------------------------------- #
# Notes
#
# 1) Wi2Vi uses 56x3x3x29 CSI, while we use 30x3x3x100
#
# 2) Video frames are aligned with the first packets of CSI
#
# 3) Wi2Vi video FPS = 30 -> 6, CSI rate = 100Hz
#
# 4) Wi2Vi train:test = 95:5
#
# 5) Wi2Vi lr=2e-3 and lower; epoch=1000; batch size=32
#
# 6) Wi2Vi outputs 320x240 images
# -------------------------------------------------------------------------- #
##############################################################################


class DropIn(nn.Module):
    def __init__(self, num_select):
        super(DropIn, self).__init__()
        self.num_select = num_select

    def forward(self, x):
        i = torch.randperm(x.shape[-1])[:self.num_select]
        return x[..., i]


class Wi2Vi(nn.Module):
    name = 'wi2vi'

    def __init__(self, batchnorm='instance'):
        super(Wi2Vi, self).__init__()

        # 56X29X18 (3x3xamp&phase)
        self.batchnorm = batchnorm
        self.Dropin = DropIn(17)
        self.EncoderOriginal = nn.Sequential(
            # 56x17x18
            nn.Conv2d(18, 64, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 56x15x64
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            # 26x7x128
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            # 12x3x256
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            # 5x1x512
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            # 5x1x512
        )

        self.Encoder = nn.Sequential(
            # 30x17x6
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 28x15x64
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            # 13x7x128
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            # 11x5x256
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            # 5x2x512
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            # 5x2x512
        )

        self.Translator_A = nn.Sequential(
            # Fill in the flattened output shape of Encoder.
            nn.Linear(5120, 972),
            nn.LeakyReLU()
        )

        self.Translator_B = nn.Sequential(
            # 36x27
            nn.ReflectionPad2d(1),
            # 38x29
            nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32x23x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 16x12x64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            # 8x6x128
        )

        self.Decoder = nn.Sequential(
            # 8x6x128
            # nn.ReflectionPad2d(1),
            # 10x8x128
            ResidualBlock(128, 128, self.batchnorm),
            # 8x6x128
            ResidualBlock(128, 128, self.batchnorm),
            # 8x6x128
            ResidualBlock(128, 128, self.batchnorm),
            # 8x6x128
            Interpolate(size=(12, 16)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            # 14x10x64
            Interpolate(size=(20, 28)),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
            # 26x18x32
            Interpolate(size=(36, 52)),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0),
            # 50x34x16
            Interpolate(size=(68, 100)),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=0),
            # 98x66x8
            Interpolate(size=(132, 196)),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=0),
            # 194x130x4
            Interpolate(size=(260, 388)),
            nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=0),
            # 386x258x2
            nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=0),
            nn.InstanceNorm2d(32),
            nn.Sigmoid()
            # 382x254x1
        )

    def forward(self, x):
        x = self.Dropin(x)
        x = self.Encoder(x)
        x = self.Translator_A(x.view(-1, 5120))
        x = self.Translator_B(x.view(-1, 1, 27, 36))
        x = self.Decoder(x)

        return x[..., 7:247, 31:351]

    def __str__(self):
        return 'Wi2Vi'


class AutoEncoder(nn.Module):
    name = 'ae'

    def __init__(self, latent_dim=16, active_func=nn.Sigmoid()):
        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.active_func = active_func

        self.EnCNN = nn.Sequential(
            nn.Conv2d(6, 128, 5, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
        )

        self.EnFC = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.latent_dim),
            nn.ReLU()
        )

        self.DeFC = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
        )
        self.DeCNN = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            self.active_func
        )

    def __str__(self):
        return f"AutoEncoder{self.latent_dim}"

    def forward(self, x):
        x = self.EnCNN(x)
        z = self.EnFC(x.view(-1, 512 * 7 * 7))
        out = self.DeFC(z)
        out = self.DeCNN(out.view(-1, 128, 4, 4))
        return z, out


class CompTrainer(BasicTrainer):
    def __init__(self, mode='wi2vi',
                 mask=False,
                 *args, **kwargs):
        super(CompTrainer, self).__init__(*args, **kwargs)

        assert mode in ('wi2vi', 'ae', 'vae', 'ae_t')

        self.mode = mode
        self.mask = mask
        self.beta = kwargs['beta'] if 'beta' in kwargs.keys() else 1
        self.modality = {'csi', 'img'}
        self.recon_lossfunc = nn.BCELoss(reduction='sum') if self.mask else nn.MSELoss(reduction='sum')

        self.loss_terms = {'LOSS'}
        self.pred_terms = ('GT', 'PRED', 'IND') if mode == 'wi2vi' else ('GT', 'PRED', 'LAT', 'IND')

        self.loss = MyLoss(loss_terms=self.loss_terms,
                           pred_terms=self.pred_terms)

    def vae_loss(self, pred, gt, mu, logvar):
        recon_loss = self.recon_lossfunc(pred, gt) / pred.shape[0]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss * self.beta
        return loss, kl_loss, recon_loss

    def calculate_loss(self, data, i=None):
        img = torch.where(data['img'] > 0, 1., 0.) if self.mask else data['img']

        if self.mode == 'wi2vi':
            output = self.models['wi2vi'](data['csi'])
            loss = self.recon_lossfunc(output, img)
            self.temp_loss = {'LOSS': loss}
            return {'GT': img,
                    'PRED': output,
                    'IND': data['ind']}

        elif self.mode == 'ae':
            latent, output = self.models['ae'](data['csi'])
            loss = self.recon_lossfunc(output, img)
            self.temp_loss = {'LOSS': loss}
            return {'GT': img,
                    'PRED': output,
                    'LAT': latent,
                    'IND': data['ind']}

        elif self.mode == 'vae':
            z, mu, logvar = self.models['csien'](data['csi'])
            output = self.models['imgde'](z)
            loss, kl_loss, recon_loss = self.vae_loss(output, img, mu, logvar)

            self.temp_loss = {'LOSS': loss,
                              'KL': kl_loss,
                              'RECON': recon_loss
                              }
            return {'GT': img,
                    'PRED': output,
                    'LAT': z,
                    'IND': data['ind']
                    }

    def plot_test(self, select_ind=None, select_num=8, autosave=False, notion='', **kwargs):
        title = {'PRED': "Test IMG Predicts",
                 'LOSS': "Test Loss"}

        filename = {term: f"{notion}_{self.name}_{term}@{self.current_ep()}.jpg" for term in ('PRED','LOSS')}

        save_path = f'../saved/{notion}/'

        if select_ind:
            inds = select_ind
        else:
            if self.inds is not None:
                inds = self.inds
            else:
                inds = self.generate_indices(self.loss.loss['pred']['IND'], select_num)

        fig1 = self.loss.plot_predict(title['PRED'], inds, ('GT', 'PRED'))
        fig3 = self.loss.plot_test(title['LOSS'], inds)

        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig1.savefig(f"{save_path}{filename['PRED']}")
            fig3.savefig(f"{save_path}{filename['LOSS']}")

        if self.mode in ('ae', 'vae'):
            title['LAT'] = "Test Latent Predicts"
            filename['LAT'] = f"{notion}_{self.name}_LAT@{self.current_ep()}.jpg"
            fig2 = self.loss.plot_latent(title['LAT'], inds, {'LAT'})
            if autosave:
                fig2.savefig(f"{save_path}{filename['LAT']}")


class CompTrainerStudent(BasicTrainer):
    def __init__(self, mask=False, alpha=0.8, *args, **kwargs):
        super(CompTrainerStudent, self).__init__(*args, **kwargs)

        self.mask = mask
        self.modality = {'csi', 'img'}

        self.alpha = alpha
        self.recon_lossfunc = nn.BCELoss(reduction='sum') if self.mask else nn.MSELoss(reduction='sum')
        self.loss_terms = ('LOSS', 'MU', 'LOGVAR', 'IMG')
        self.pred_terms = ('GT', 'T_PRED', 'S_PRED', 'T_LATENT', 'S_LATENT', 'IND')
        self.loss = MyLoss(loss_terms=self.loss_terms,
                           pred_terms=self.pred_terms)

    def kd_loss(self, mu_s, logvar_s, mu_t, logvar_t):
        mu_loss = self.recon_lossfunc(mu_s, mu_t) / mu_s.shape[0]
        logvar_loss = self.recon_lossfunc(logvar_s, logvar_t) / logvar_s.shape[0]
        loss = self.alpha * mu_loss + (1 - self.alpha) * logvar_loss
        return loss, mu_loss, logvar_loss

    def calculate_loss(self, data):
        img = torch.where(data['img'] > 0, 1., 0.) if self.mask else data['img']

        s_z, s_mu, s_logvar = self.models['csien'](data['csi'])

        with torch.no_grad():
            t_z, t_mu, t_logvar = self.models['imgen'](img)
            s_output = self.models['imgde'](s_z)
            t_output = self.models['imgde'](t_z)
            image_loss = self.recon_lossfunc(s_output, img)

        loss_i, mu_loss_i, logvar_loss_i = self.kd_loss(s_mu, s_logvar, t_mu, t_logvar)
        loss = loss_i

        self.temp_loss = {'LOSS': loss,
                          'MU': mu_loss_i,
                          'LOGVAR': logvar_loss_i,
                          'IMG': image_loss}
        return {'GT': img,
                'T_LATENT': torch.cat((t_mu, t_logvar), -1),
                'S_LATENT': torch.cat((s_mu, s_logvar), -1),
                'T_PRED': t_output,
                'S_PRED': s_output,
                'IND': data['ind']}

    def plot_test(self, select_ind=None, select_num=8, autosave=False, notion='', **kwargs):
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


if __name__ == "__main__":
    #m1 = Wi2Vi()
    #summary(m1, input_size=(6, 30, 30))
    m2 = AutoEncoderNew()
    summary(m2, input_size=(6, 30, 30))
