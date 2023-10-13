import torch
import torch.nn as nn
import numpy as np
import matplotlib.ticker as ticker
from scipy.stats import norm
import matplotlib.pyplot as plt
from TrainerTS import timer, MyDataset, split_loader, MyArgs, TrainerTS

# ------------------------------------- #
# Trainer of VAE Teacher-student network


class TrainerVTS(TrainerTS):
    def __init__(self, img_encoder, img_decoder, csi_encoder,
                 teacher_args, student_args,
                 train_loader, valid_loader, test_loader,
                 div_loss=nn.KLDivLoss(reduction='batchmean'),
                 img_loss=nn.MSELoss(reduction='sum'),
                 temperature=20,
                 alpha=0.3,
                 latent_dim=16,
                 kl_weight=1.2
                 ):
        super(TrainerVTS, self).__init__(img_encoder=img_encoder, img_decoder=img_decoder, csi_encoder=csi_encoder,
                                         teacher_args=teacher_args, student_args=student_args,
                                         train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
                                         div_loss=div_loss,
                                         img_loss=img_loss,
                                         temperature=temperature,
                                         alpha=alpha,
                                         latent_dim=latent_dim)
        self.kl_weight = kl_weight

    @staticmethod
    def __gen_teacher_train__():
        t_train_loss = {'learning_rate': [],
                        'epochs': [0],
                        'LOSS': [],
                        'KL': [],
                        'RECON': [],
                        }
        return t_train_loss

    @staticmethod
    def __gen_teacher_test__():
        t_test_loss = {'LOSS': [],
                       'RECON': [],
                       'KL': [],
                       'PRED': [],
                       'GT': [],
                       'IND': []}
        return t_test_loss

    @staticmethod
    def __teacher_plot_terms__():
        terms = {'loss': {'LOSS': 'Loss',
                          'KL': 'KL Loss',
                          'RECON': 'Reconstruction Loss'
                          },
                 'predict': ('GT', 'PRED', 'IND'),
                 'test': {'GT': 'Ground Truth',
                          'PRED': 'Estimated'}
                 }
        return terms

    @staticmethod
    def kl_loss(vector):
        mu = vector[:len(vector)//2]
        logvar = vector[len(vector)//2:]
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def loss(self, y, gt, latent):
        # reduction = 'sum'
        recon_loss = self.args['t'].criterion(y, gt) / y.shape[0]
        kl_loss = self.kl_loss(latent)
        loss = recon_loss + kl_loss * self.kl_weight
        return loss, kl_loss, recon_loss

    def calculate_loss_t(self, x, y, i=None):

        latent, z = self.models['imgen'](y)
        output = self.models['imgde'](z)
        loss, kl_loss, recon_loss = self.loss(output, y, latent)
        self.temp_loss = {'LOSS': loss,
                          'KL': kl_loss,
                          'RECON': recon_loss}
        return {'GT': y,
                'PRED': output,
                'IND': i}

    def calculate_loss_s(self, x, y, i=None):

        s_latent, s_z = self.models['csien'](x)
        with torch.no_grad():
            t_latent, t_z = self.models['imgen'](y)
            s_output = self.models['imgde'](s_z)
            t_output = self.models['imgde'](t_z)
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
                'S_LATENT': s_latent,
                'T_PRED': t_output,
                'S_PRED': s_output,
                'IND': i}

    def traverse_latent_2dim(self, img_ind, dataset, mode='t', img='x',
                             dim1=0, dim2=1, granularity=11, autosave=False, notion=''):
        self.__plot_settings__()
        self.__test_models_s__()

        if img_ind >= len(dataset):
            img_ind = np.random.randint(len(dataset))

        try:
            data_x, data_y, index = dataset.__getitem__(img_ind)
            if img == 'x':
                image = data_x[np.newaxis, ...]
            elif img == 'y':
                image = data_y[np.newaxis, ...]
                csi = data_x[np.newaxis, ...]

        except ValueError:
            image = dataset[img_ind][np.newaxis, ...]

        if mode == 't':
            latent, z = self.models['imgen'](torch.from_numpy(image).to(torch.float32).to(self.args['t'].device))
        elif mode == 's':
            latent, z = self.models['csien'](torch.from_numpy(csi).to(torch.float32).to(self.args['s'].device))

        e = z.cpu().detach().numpy().squeeze()

        grid_x = norm.ppf(np.linspace(0.05, 0.95, granularity))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, granularity))
        anchor1 = np.searchsorted(grid_x, e[dim1])
        anchor2 = np.searchsorted(grid_y, e[dim2])
        anchor1 = anchor1 * 128 if anchor1 < granularity else (anchor1 - 1) * 128
        anchor2 = anchor2 * 128 if anchor2 < granularity else (anchor2 - 1) * 128

        figure = np.zeros((granularity * 128, granularity * 128))

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                e[dim1], e[dim2] = xi, yi
                output = self.models['imgde'](torch.from_numpy(e).to(torch.float32).to(self.args['t'].device))
                figure[i * 128: (i + 1) * 128,
                       j * 128: (j + 1) * 128] = output.cpu().detach().numpy().squeeze().tolist()

        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Teacher Traverse in dims {dim1}_{dim2}")
        plt.imshow(figure)
        rect = plt.Rectangle((anchor1, anchor2), 128, 128, fill=False, edgecolor='orange')
        ax = plt.gca()
        ax.add_patch(rect)
        plt.axis('off')
        plt.xlabel(str(dim1))
        plt.ylabel(str(dim2))

        if autosave:
            plt.savefig(f"{self.current_title()}_T_traverse_{dim1}{dim2}_{notion}.jpg")
        plt.show()

    def traverse_latent(self, img_ind, dataset, mode='t', img='x',  autosave=False, notion=''):
        self.__plot_settings__()
        self.__test_models_s__()

        if img_ind >= len(dataset):
            img_ind = np.random.randint(len(dataset))

        try:
            data_x, data_y, index = dataset.__getitem__(img_ind)
            if img == 'x':
                image = data_x[np.newaxis, ...]
            elif img == 'y':
                image = data_y[np.newaxis, ...]
                csi = data_x[np.newaxis, ...]

        except ValueError:
            image = dataset[img_ind][np.newaxis, ...]

        if mode == 't':
            latent, z = self.models['imgen'](torch.from_numpy(image).to(torch.float32).to(self.args['t'].device))
        elif mode == 's':
            latent, z = self.models['csien'](torch.from_numpy(csi).to(torch.float32).to(self.args['s'].device))

        e = z.cpu().detach().numpy().squeeze()

        figure = np.zeros((self.latent_dim * 128, self.latent_dim * 128))

        anchors = []
        for dim in range(self.latent_dim):
            grid_x = norm.ppf(np.linspace(0.05, 0.95, self.latent_dim))
            anchor = np.searchsorted(grid_x, e[dim])
            anchors.append(anchor * 128 if anchor < self.latent_dim else (anchor - 1) * 128)

            for i in range(self.latent_dim):
                for j, xi in enumerate(grid_x):
                    e[dim] = xi
                    output = self.models['imgde'](torch.from_numpy(e).to(torch.float32).to(self.args['t'].device))
                    figure[i * 128: (i + 1) * 128,
                           j * 128: (j + 1) * 128] = output.cpu().detach().numpy().squeeze().tolist()

        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Teacher Traverse in dims 0~{self.latent_dim - 1}")
        plt.imshow(figure)
        for i, an in enumerate(anchors):
            rect = plt.Rectangle((an, i * 128), 128, 128, fill=False, edgecolor='orange')
            ax = plt.gca()
            ax.add_patch(rect)
        # plt.axis('off')
        plt.xticks([x * 128 for x in (range(self.latent_dim))], [x for x in (range(self.latent_dim))])
        plt.yticks([x * 128 for x in (range(self.latent_dim))], [x for x in (range(self.latent_dim))])
        plt.xlabel('Traversing')
        plt.ylabel('Dimensions')

        if autosave:
            plt.savefig(f"{notion}_T_traverse_{self.latent_dim}_{self.current_title()}.jpg")
        plt.show()


class TrainerVTSMask(TrainerVTS):
    def __init__(self, img_encoder, img_decoder, csi_encoder, msk_decoder,
                 teacher_args, student_args,
                 train_loader, valid_loader, test_loader,
                 ):
        super(TrainerVTSMask, self).__init__(img_encoder=img_encoder, img_decoder=img_decoder, csi_encoder=csi_encoder,
                                            teacher_args=teacher_args, student_args=student_args,
                                            train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
         )
        self.mask_loss = nn.BCELoss(reduction='sum')
        self.models['mskde'] = msk_decoder

    @staticmethod
    def __gen_teacher_train__():
        t_train_loss = {'learning_rate': [],
                        'epochs': [0],
                        'LOSS': [],
                        'KL': [],
                        'RECON': [],
                        'MASK': [],
                        }

        return t_train_loss

    @staticmethod
    def __gen_teacher_test__():
        t_test_loss = {'LOSS': [],
                       'RECON': [],
                       'KL': [],
                       'MASK': [],
                       'PRED': [],
                       'PRED_MASK': [],
                       'GT': [],
                       'GT_MASK': [],
                       'IND': []}
        return t_test_loss

    @staticmethod
    def __teacher_plot_terms__():
        terms = {'loss': {'LOSS': 'Loss',
                          'KL': 'KL Loss',
                          'RECON': 'Reconstruction Loss',
                          'MASK': 'Mask Loss'
                          },
                 'predict': ('GT', 'GT_MASK', 'PRED', 'PRED_MASK', 'IND'),
                 'test': {'GT': 'Ground Truth',
                          'GT_MASK': 'GT Mask',
                          'PRED': 'Estimated',
                          'PRED_MASK': 'Estimated Mask'
                          }
                 }
        return terms

    def __train_models_t__(self):
        self.models['imgen'].train()
        self.models['imgde'].train()
        return [{'params': self.models['imgen'].parameters()},
                {'params': self.models['imgde'].parameters()},
                {'params': self.models['mskde'].parameters()}]

    def __test_models_t__(self):
        self.models['imgen'].eval()
        self.models['imgde'].eval()
        self.models['mskde'].eval()

    def loss(self, y, m, gt_y, gt_m, latent):
        # reduction = 'sum'
        recon_loss = self.args['t'].criterion(y, gt_y) / y.shape[0]
        kl_loss = self.kl_loss(latent)
        mask_loss = self.mask_loss(m, gt_m) / m.shape[0]
        # loss = recon_loss + kl_loss * self.kl_weight + mask_loss
        loss = mask_loss
        return loss, kl_loss, recon_loss, mask_loss

    def calculate_loss_t(self, x, y, i=None):

        gt_mask = torch.where(y > 0, 1., 0.)

        latent, z = self.models['imgen'](y)
        output = self.models['imgde'](z)
        mask = self.models['mskde'](z)
        # output = output.mul(mask)
        loss, kl_loss, recon_loss, mask_loss = self.loss(output, mask, y, gt_mask, latent)
        self.temp_loss = {'LOSS': loss,
                          'KL': kl_loss,
                          'RECON': recon_loss,
                          'MASK': mask_loss}
        return {'GT': y,
                'GT_MASK': gt_mask,
                'PRED': output,
                'PRED_MASK': mask,
                'IND': i}


