import numpy as np
import matplotlib.pyplot as plt
from Trainer_v03b3 import ImageEncoderV03b3, ImageDecoderV03b3
from ModelVTS import *
from TrainerTS import timer, MyDataset, split_loader, MyArgs, bn, Interpolate
from TrainerVTS import TrainerVTS

# ------------------------------------- #
# Model v03b5
# Minor modifications to Model v03b3
# Cycle-consistent training

# ImageEncoder: in = 128 * 128, out = 1 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128
# CSIEncoder: in = 2 * 90 * 100, out = 1 * 256 (Unused)


class TrainerVTSM1(TrainerVTS):
    def __init__(self, *args, **kwargs):
        super(TrainerVTSM1, self).__init__(*args, **kwargs)

    @staticmethod
    def __teacher_plot_terms__():
        terms = {'loss': {'LOSS': 'Loss',
                          'KL': 'KL Loss',
                          'RECON': 'Reconstruction Loss',
                          'CYCLE': 'Cycle Consistency Loss'
                          },
                 'predict': ('GT', 'PRED', 'RE_PRED', 'IND'),
                 'test': {'GT': 'Ground Truth',
                          'PRED': 'Estimated',
                          'RE_PRED': 'Re-Estimated'}
                 }
        return terms

    @staticmethod
    def __gen_teacher_train__():
        t_train_loss = {'learning_rate': [],
                        'epochs': [0],
                        'LOSS': [],
                        'KL': [],
                        'RECON': [],
                        'CYCLE': [],
                        }

        return t_train_loss

    @staticmethod
    def __gen_teacher_test__():
        t_test_loss = {'LOSS': [],
                       'RECON': [],
                       'KL': [],
                       'CYCLE': [],
                       'PRED': [],
                       'RE_PRED': [],
                       'GT': [],
                       'IND': []}
        return t_test_loss

    def loss(self, y, gt, latent, latent_p):
        # set reduce = 'sum'
        # considering batch
        recon_loss = self.args['t'].criterion(y, gt) / y.shape[0]
        latent_loss = self.args['s'].criterion(latent_p, latent) / y.shape[0]
        kl_loss = self.kl_loss(latent)
        loss = recon_loss + kl_loss * self.kl_weight + latent_loss
        return loss, kl_loss, recon_loss, latent_loss

    def calculate_loss(self, mode, x, y, i=None):
        if mode == 't':
            latent, z = self.img_encoder(y)
            output = self.img_decoder(z)
            with torch.no_grad():
                latent_r, z_r = self.img_encoder(output)
                re_output = self.img_decoder(z_r)
            loss, kl_loss, recon_loss, latent_loss = self.loss(output, y, latent, latent_r)
            self.temp_loss = {'LOSS': loss,
                              'KL': kl_loss,
                              'RECON': recon_loss,
                              'CYCLE': latent_loss}
            return {'GT': y,
                    'PRED': output,
                    'RE_PRED': re_output,
                    'IND': i}

        elif mode == 's':
            s_latent, s_z = self.csi_encoder(x)
            with torch.no_grad():
                t_latent, t_z = self.img_encoder(y)
                s_output = self.img_decoder(s_z)
                t_output = self.img_decoder(t_z)
                image_loss = self.img_loss(s_output, y)

            straight_loss = self.args['s'].criterion(s_latent, t_latent)
            distil_loss = self.div_loss(self.logsoftmax(s_latent / self.temperature),
                                        nn.functional.softmax(t_latent / self.temperature, -1))
            loss = self.alpha * straight_loss + (1 - self.alpha) * distil_loss
            self.temp_loss = {'LOSS': loss,
                              'STRA': straight_loss,
                              'DIST': distil_loss,
                              'IMG': image_loss}
            return {'GT': y,
                    'T_LATENT': t_latent,
                    'LATENT': s_latent,
                    'T_PRED': t_output,
                    'PRED': s_output,
                    'IND': i}
