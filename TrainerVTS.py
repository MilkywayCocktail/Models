import torch
import torch.nn as nn
import numpy as np
import matplotlib.ticker as ticker
from scipy.stats import norm
import matplotlib.pyplot as plt
from TrainerTS import timer, MyArgs, TrainerTS

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
        """
        Generates student's training loss.
        :return: structured loss
        """
        t_train_loss = {'learning_rate': [],
                        'epochs': [0],
                        'LOSS': [],
                        'KL': [],
                        'RECON': [],
                        }
        return t_train_loss

    @staticmethod
    def __gen_teacher_test__():
        """
        Generates teacher's test loss.
        :return: structured loss
        """
        t_test_loss = {'LOSS': [],
                       'RECON': [],
                       'KL': [],
                       'PRED': [],
                       'GT': [],
                       'IND': []}
        return t_test_loss

    @staticmethod
    def __teacher_plot_terms__():
        """
        Defines plot items for plot_test(mode='t')
        :return: keywords
        """
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
    def kl_loss(mu, logvar):
        """
        KL loss used as VAE loss.
        :param mu: mu vector
        :param logvar: sigma vector
        :return: loss term
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def vae_loss(self, pred, gt, mu, logvar):
        """
        Total loss of teacher network.\n
        :param pred: predicted images
        :param gt: ground truth of images
        :param mu: predicted mu vector
        :param logvar: predicted logvar vector
        :return: loss term
        """
        # reduction = 'sum'
        recon_loss = self.args['t'].criterion(pred, gt) / pred.shape[0]
        kl_loss = self.kl_loss(mu, logvar)
        loss = recon_loss + kl_loss * self.kl_weight
        return loss, kl_loss, recon_loss

    def kd_loss(self, s_latent, t_latent):
        straight_loss = self.args['s'].criterion(s_latent, t_latent) / s_latent.shape[0]
        distil_loss = self.div_loss(self.logsoftmax(s_latent / self.temperature),
                                    nn.functional.softmax(t_latent / self.temperature, -1))
        loss = self.alpha * straight_loss + (1 - self.alpha) * distil_loss
        return loss, straight_loss, distil_loss

    def calculate_loss_t(self, x, y, i=None):
        """
        Calculates loss function for back propagation,
        :param x: x data (CSI)
        :param y: y data (image)
        :param i: index of data
        :return: loss object
        """
        latent, z, mu, logvar = self.models['imgen'](y)
        output = self.models['imgde'](z)
        loss, kl_loss, recon_loss = self.vae_loss(output, y, mu, logvar)
        self.temp_loss = {'LOSS': loss,
                          'KL': kl_loss,
                          'RECON': recon_loss}
        return {'GT': y,
                'PRED': output,
                'IND': i}

    def calculate_loss_s(self, x, y, i=None):
        """
        Calculates loss function for back propagation.
        :param x: x data (CSI)
        :param y: y data (image)
        :param i: index of data
        :return: loss object
        """
        s_latent, s_z, s_mu, s_logvar = self.models['csien'](x)
        with torch.no_grad():
            t_latent, t_z, t_mu, t_logvar = self.models['imgen'](y)
            s_output = self.models['imgde'](s_z)
            t_output = self.models['imgde'](t_z)
            image_loss = self.img_loss(s_output, y)
        loss, straight_loss, distil_loss = self.kd_loss(s_latent, t_latent)

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
        """
        Generates student's training loss.
        :return: structured loss
        """
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
        """
        Generates teacher's test loss.
        :return: structured loss
        """
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
        """
        Defines plot items for plot_test(mode='t')
        :return: keywords
        """
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
        """
        Changes teacher model states for training.
        :return: None
        """
        self.models['imgen'].train()
        self.models['imgde'].train()
        return [{'params': self.models['imgen'].parameters()},
                {'params': self.models['imgde'].parameters()},
                {'params': self.models['mskde'].parameters()}]

    def __test_models_t__(self):
        """
        Changes teacher model states for testing.
        :return: None
        """
        self.models['imgen'].eval()
        self.models['imgde'].eval()
        self.models['mskde'].eval()

    def loss(self, y, pred_mask, gt_y, gt_m, latent):
        """
        Total loss of teacher network.
        :param y: predicted images
        :param pred_mask: predicted masks
        :param gt_y: ground truth of images
        :param gt_m: ground truth of masks
        :param latent: predicted latent vectors
        :return: loss term
        """

        # reduction = 'sum'
        recon_loss = self.args['t'].criterion(y, gt_y) / y.shape[0]
        kl_loss = self.kl_loss(latent)
        mask_loss = self.mask_loss(pred_mask, gt_m) / pred_mask.shape[0]
        loss = recon_loss + kl_loss * self.kl_weight + mask_loss
        # loss = mask_loss
        return loss, kl_loss, recon_loss, mask_loss

    def calculate_loss_t(self, x, y, i=None):
        """
        Calculates loss function for back propagation,
        :param x: x data (CSI)
        :param y: y data (image)
        :param i: index of data
        :return: loss object
        """
        gt_mask = torch.where(y > 0, 1., 0.)

        latent, z = self.models['imgen'](y)
        output = self.models['imgde'](z)
        mask = self.models['mskde'](z)
        output = output.mul(mask)
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


class TrainerVTSIB1(TrainerVTS):
    def __init__(self, img_encoder, img_decoder, csi_encoder,
                 teacher_args, student_args,
                 train_loader, valid_loader, test_loader,
                 lb=1e-2,
                 inductive_length=25,
                 memory_size=64
                 ):
        super(TrainerVTSIB1, self).__init__(img_encoder=img_encoder, img_decoder=img_decoder, csi_encoder=csi_encoder,
                                            teacher_args=teacher_args, student_args=student_args,
                                            train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
         )

        self.lb = lb
        self.inductive_length = inductive_length
        self.memory_size = memory_size
        self.ib_loss = nn.MSELoss()
        self.memory_bank = self.__gen_memory_bank__()

    @staticmethod
    def __gen_teacher_train__():
        """
        Generates teacher's training loss.
        :return: structured loss
        """
        t_train_loss = {'learning_rate': [],
                        'epochs': [0],
                        'LOSS': [],
                        'KL': [],
                        'RECON': [],
                        'ORTH': [],
                        'SMTH': []
                        }

        return t_train_loss

    @staticmethod
    def __gen_teacher_test__():
        """
        Generates teacher's test loss.
        :return: structured loss
        """
        t_test_loss = {'LOSS': [],
                       'RECON': [],
                       'KL': [],
                       'PRED': [],
                       'GT': [],
                       'GT_IB': [],
                       'IND': []}
        return t_test_loss

    @staticmethod
    def __teacher_plot_terms__():
        """
        Defines plot items for plot_test(mode='t')
        :return: keywords
        """
        terms = {'loss': {'LOSS': 'Loss',
                          'KL': 'KL Loss',
                          'RECON': 'Reconstruction Loss',
                          'ORTH': 'Orthogonality Loss',
                          'SMTH': 'Smoothness Loss'
                          },
                 'predict': ('GT', 'PRED', 'IND'),
                 'test': {'GT': 'Ground Truth',
                          'PRED': 'Estimated',
                          }
                 }
        return terms

    def __gen_memory_bank__(self):
        """
        Generates memory bank.
        :return: memory bank
        """
        mem = {'LAT': torch.zeros(self.memory_size),
               'IB': torch.zeros(self.memory_size),
               'FLAG': 0
               }
        return mem

    def orthogonality_loss(self, ib):
        """
        Orthogonality loss upon inductive bias.
        :param ib: predicted inductive bias
        :return: loss term
        """
        ib_dims = ib.view(-1, 5, 5)
        orth_matrix = ib_dims * torch.transpose(ib_dims, -1, -2)    # torch.matmul
        orth_loss = self.ib_loss(orth_matrix, torch.eye(5))

        return orth_loss

    def smoothness_loss(self, latent, gt_ib):
        if self.memory_bank['FLAG'] == 0:
            self.memory_bank['LAT'] = latent
            self.memory_bank['IB'] = gt_ib
            self.memory_bank['FLAG'] = 1
            return 0
        else:
            d_y = gt_ib - self.memory_bank['IB']
            d_x = latent - self.memory_bank['LAT']
            gradient2 = (torch.matmul(d_y, torch.transpose(d_y, -1, -2))) / torch.matmul(d_x, torch.matmul(d_x))
            smooth_loss = self.ib_loss(gradient2, self.lb * self.lb)
            self.memory_bank['LAT'] = latent
            self.memory_bank['IB'] = gt_ib
        return smooth_loss

    def loss(self, pred, pred_ib, gt_y, gt_ib, latent):
        """
        Total loss of teacher network.
        :param pred: predicted images
        :param pred_ib: predicted inductive biases
        :param gt_y: ground truth of images
        :param gt_ib: ground truth of inductive biases
        :param latent: predicted latent vectors
        :return: loss term
        """
        # reduction = 'sum'
        recon_loss = self.args['t'].criterion(pred, gt_y) / pred.shape[0]
        kl_loss = self.kl_loss(latent)
        ib_loss1 = self.orthogonality_loss(pred_ib)
        ib_loss2 = self.smoothness_loss(latent, gt_ib)

        loss = recon_loss + kl_loss * self.kl_weight + ib_loss1 + ib_loss2

        return loss, kl_loss, recon_loss, ib_loss1, ib_loss2

    def calculate_loss_t(self, x, y, c, i=None):
        """
        Calculates loss function for back propagation,
        :param x: x data (CSI)
        :param y: y data (image)
        :param c: inductive bias data
        :param i: index of data
        :return: loss object
        """
        latent, z = self.models['imgen'](y)
        output, ib = self.models['imgde'](z)

        loss, kl_loss, recon_loss, orth_loss, smooth_loss = self.loss(output, ib, y, c, latent)
        self.temp_loss = {'LOSS': loss,
                          'KL': kl_loss,
                          'RECON': recon_loss,
                          'ORTH': orth_loss,
                          'SMTH': smooth_loss}
        return {'GT': y,
                'GT_IB': c,
                'PRED': output,
                'IND': i}


class TrainerVTSMu(TrainerVTS):
    def __init__(self, img_encoder, img_decoder, csi_encoder,
                 teacher_args, student_args,
                 train_loader, valid_loader, test_loader,
                 alpha=0.3,
                 latent_dim=16,
                 kl_weight=1.2
                 ):
        super(TrainerVTSMu, self).__init__(img_encoder=img_encoder, img_decoder=img_decoder, csi_encoder=csi_encoder,
                                           teacher_args=teacher_args, student_args=student_args,
                                           train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
                                           alpha=alpha,
                                           latent_dim=latent_dim,
                                           kl_weight=kl_weight
        )

    @staticmethod
    def __gen_student_train__():
        """
        Generates student's training loss.
        :return: structured loss
        """
        s_train_loss = {'learning_rate': [],
                        'epochs': [0],
                        'LOSS': [],
                        'MU': [],
                        'LOGVAR': [],
                        'IMG': [],
                        }
        return s_train_loss

    @staticmethod
    def __gen_student_test__():
        """
        Generates student's test loss.
        :return: structured loss
        """
        s_test_loss = {'LOSS': [],
                       'MU': [],
                       'LOGVAR': [],
                       'IMG': [],
                       'T_LATENT': [],
                       'S_LATENT': [],
                       'T_PRED': [],
                       'S_PRED': [],
                       'GT': [],
                       'IND': []
                       }
        return s_test_loss

    @staticmethod
    def __student_plot_terms__():
        """
        Defines plot items for plot_test(mode='s')
        :return: keywords
        """
        terms = {'loss': {'LOSS': 'Loss',
                          'MU': 'Mu',
                          'LOGVAR': 'Logvar',
                          'IMG': 'Image'},
                 'predict': ('GT', 'S_PRED', 'T_PRED', 'T_LATENT', 'S_LATENT', 'IND'),
                 'test': {'GT': 'Ground Truth',
                          'T_PRED': 'Teacher Estimate',
                          'S_PRED': 'Student Estimate'}
                 }
        return terms

    def kd_loss(self, mu_s, logvar_s, mu_t, logvar_t):
        mu_loss = self.args['s'].criterion(mu_s, mu_t) / mu_s.shape[0]
        logvar_loss = self.args['s'].criterion(logvar_s, logvar_t) / logvar_s.shape[0]
        loss = self.alpha * mu_loss + (1 - self.alpha) * logvar_loss
        return loss, mu_loss, logvar_loss

    def calculate_loss_s(self, x, y, i=None):
        """
        Calculates loss function for back propagation.
        :param x: x data (CSI)
        :param y: y data (image)
        :param i: index of data
        :return: loss object
        """
        s_latent, s_z, s_mu, s_logvar = self.models['csien'](x)
        with torch.no_grad():
            t_latent, t_z, t_mu, t_logvar = self.models['imgen'](y)
            s_output = self.models['imgde'](s_z)
            t_output = self.models['imgde'](t_z)
            image_loss = self.img_loss(s_output, y)
        loss, mu_loss, logvar_loss = self.kd_loss(s_mu, s_logvar, t_mu, t_logvar)

        self.temp_loss = {'LOSS': loss,
                          'MU': mu_loss,
                          'LOGVAR': logvar_loss,
                          'IMG': image_loss}
        return {'GT': y,
                'T_LATENT': t_latent,
                'S_LATENT': s_latent,
                'T_PRED': t_output,
                'S_PRED': s_output,
                'IND': i}


