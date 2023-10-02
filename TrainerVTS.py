import torch
import torch.nn as nn
import numpy as np
import matplotlib.ticker as ticker
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
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
                 latent_dim=8,
                 kl_weight=0.25
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

    def calculate_loss(self, mode, x, y, i=None):
        if mode == 't':
            latent, z = self.img_encoder(y)
            output = self.img_decoder(z)
            loss, kl_loss, recon_loss = self.loss(output, y, latent)
            self.temp_loss = {'LOSS': loss,
                              'KL': kl_loss,
                              'RECON': recon_loss}
            return {'GT': y,
                    'PRED': output,
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
                    'S_LATENT': s_latent,
                    'T_PRED': t_output,
                    'S_PRED': s_output,
                    'IND': i}

    def traverse_latent_2dim(self, img_ind, dataset, mode='t', img='x',
                             dim1=0, dim2=1, granularity=11, autosave=False, notion=''):
        self.__plot_settings__()

        self.img_encoder.eval()
        self.img_decoder.eval()
        self.csi_encoder.eval()

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
            latent, z = self.img_encoder(torch.from_numpy(image).to(torch.float32).to(self.args['t'].device))
        elif mode == 's':
            latent, z = self.csi_encoder(torch.from_numpy(csi).to(torch.float32).to(self.args['s'].device))

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
                output = self.img_decoder(torch.from_numpy(e).to(torch.float32).to(self.args['t'].device))
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

        self.img_encoder.eval()
        self.img_decoder.eval()
        self.csi_encoder.eval()

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
            latent, z = self.img_encoder(torch.from_numpy(image).to(torch.float32).to(self.args['t'].device))
        elif mode == 's':
            latent, z = self.csi_encoder(torch.from_numpy(csi).to(torch.float32).to(self.args['s'].device))

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
                    output = self.img_decoder(torch.from_numpy(e).to(torch.float32).to(self.args['t'].device))
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
                 div_loss=nn.KLDivLoss(reduction='batchmean'),
                 img_loss=nn.MSELoss(reduction='sum'),
                 temperature=20,
                 alpha=0.3,
                 latent_dim=8,
                 kl_weight=0.25
                 ):
        super(TrainerVTSMask, self).__init__(img_encoder=img_encoder, img_decoder=img_decoder, csi_encoder=csi_encoder,
                                         teacher_args=teacher_args, student_args=student_args,
                                         train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
                                         div_loss=div_loss,
                                         img_loss=img_loss,
                                         temperature=temperature,
                                         alpha=alpha,
                                         latent_dim=latent_dim)
        self.kl_weight = kl_weight
        self.mask_loss = nn.BCELoss(reduction='sum')
        self.msk_decoder = msk_decoder

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
                       'PRED': [],
                       'MASK': [],
                       'GT': [],
                       'IND': []}
        return t_test_loss

    @staticmethod
    def __teacher_plot_terms__():
        terms = {'loss': {'LOSS': 'Loss',
                          'KL': 'KL Loss',
                          'RECON': 'Reconstruction Loss',
                          'MASK': 'Mask Loss'
                          },
                 'predict': ('GT', 'PRED', 'MASK', 'IND'),
                 'test': {'GT': 'Ground Truth',
                          'PRED': 'Estimated',
                          'MASK': "Est Mask"}
                 }
        return terms

    def loss(self, y, m, gt_y, gt_m, latent):
        # reduction = 'sum'
        recon_loss = self.args['t'].criterion(y, gt_y) / y.shape[0]
        kl_loss = self.kl_loss(latent)
        mask_loss = self.mask_loss(m, gt_m)
        loss = recon_loss + kl_loss * self.kl_weight + mask_loss
        return loss, kl_loss, recon_loss, mask_loss

    def calculate_loss(self, mode, x, y, i=None):
        if mode == 't':
            latent, z = self.img_encoder(y)
            output = self.img_decoder(z)
            mask = self.msk_decoder(z)
            one = torch.ones_like(y)
            gt_mask = torch.where(y > 0, one, y)
            loss, kl_loss, recon_loss, mask_loss = self.loss(output, mask, y, gt_mask, latent)
            self.temp_loss = {'LOSS': loss,
                              'KL': kl_loss,
                              'RECON': recon_loss,
                              'MASK': mask_loss}
            return {'GT': y,
                    'PRED': output,
                    'MASK': mask,
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
                    'S_LATENT': s_latent,
                    'T_PRED': t_output,
                    'S_PRED': s_output,
                    'IND': i}

    @timer
    def train_teacher(self, autosave=False, notion=''):
        """
        Trains the teacher.
        :param autosave: whether to save model parameters. Default is False
        :param notion: additional notes in save name
        :return: trained teacher
        """
        self.logger(mode='t')
        teacher_optimizer = self.args['t'].optimizer([{'params': self.img_encoder.parameters()},
                                                      {'params': self.img_decoder.parameters()}],
                                                     lr=self.args['t'].learning_rate)
        LOSS_TERMS = self.plot_terms['t']['loss'].keys()

        for epoch in range(self.args['t'].epochs):

            # =====================train============================
            self.img_encoder.train()
            self.img_decoder.train()
            self.msk_decoder.train()
            EPOCH_LOSS = {key: [] for key in LOSS_TERMS}
            for idx, (data_x, data_y, index) in enumerate(self.train_loader, 0):
                data_y = data_y.to(torch.float32).to(self.args['t'].device)
                teacher_optimizer.zero_grad()

                PREDS = self.calculate_loss(mode='t', x=None, y=data_y)
                self.temp_loss['LOSS'].backward()
                teacher_optimizer.step()
                for key in LOSS_TERMS:
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

                if idx % (len(self.train_loader) // 5) == 0:
                    print(f"\rTeacher: epoch={epoch}/{self.args['t'].epochs}, batch={idx}/{len(self.train_loader)},"
                          f"loss={self.temp_loss['LOSS'].item()}", end='')
            for key in LOSS_TERMS:
                self.train_loss['t'][key].append(np.average(EPOCH_LOSS[key]))

            # =====================valid============================
            self.img_encoder.eval()
            self.img_decoder.eval()
            EPOCH_LOSS = {key: [] for key in LOSS_TERMS}

            for idx, (data_x, data_y, index) in enumerate(self.valid_loader, 0):
                data_y = data_y.to(torch.float32).to(self.args['t'].device)
                with torch.no_grad():
                    PREDS = self.calculate_loss('t', None, data_y)
                for key in LOSS_TERMS:
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())
            for key in LOSS_TERMS:
                self.valid_loss['t'][key].append(np.average(EPOCH_LOSS[key]))

        if autosave:
            save_path = f'../saved/{notion}/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.img_encoder.state_dict(),
                       f"{save_path}{notion}_{self.img_encoder}_{self.current_title()}.pth")
            torch.save(self.img_decoder.state_dict(),
                       f"{save_path}{notion}_{self.img_decoder}_{self.current_title()}.pth")

    def test_teacher(self, mode='test'):
        """
        Tests the teacher and saves estimates.
        :param mode: 'test' or 'train' (selects data loader). Default is 'test'
        :return: test results
        """
        self.test_loss['t'] = self.__gen_teacher_test__()
        self.img_encoder.eval()
        self.img_decoder.eval()
        self.msk_decoder.eval()
        LOSS_TERMS = self.plot_terms['t']['loss'].keys()
        PLOT_TERMS = self.plot_terms['t']['predict']
        EPOCH_LOSS = {key: [] for key in LOSS_TERMS}

        if mode == 'test':
            loader = self.test_loader
        elif mode == 'train':
            loader = self.train_loader

        for idx, (data_x, data_y, index) in enumerate(loader, 0):
            data_y = data_y.to(torch.float32).to(self.args['t'].device)
            with torch.no_grad():
                for sample in range(loader.batch_size):
                    ind = index[sample][np.newaxis, ...]
                    gt = data_y[sample][np.newaxis, ...]
                    PREDS = self.calculate_loss(mode='t', x=None, y=gt, i=ind)

                    for key in LOSS_TERMS:
                        EPOCH_LOSS[key].append(self.temp_loss[key].item())

                    for key in PLOT_TERMS:
                        self.test_loss['t'][key].append(PREDS[key].cpu().detach().numpy().squeeze())
                    for key in LOSS_TERMS:
                        self.test_loss['t'][key] = EPOCH_LOSS[key]

            if idx % (len(loader)//5) == 0:
                print(f"\rTeacher: test={idx}/{len(loader)}, loss={self.temp_loss['LOSS'].item()}", end='')

        for key in LOSS_TERMS:
            EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
        print(f"\nTest finished. Average loss={EPOCH_LOSS}")