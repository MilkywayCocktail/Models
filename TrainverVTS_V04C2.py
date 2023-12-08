import torch
import torch.nn as nn
from torchvision.ops import generalized_box_iou_loss
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from Loss import MyLoss_T_BBX, MyLoss_S_BBX


def timer(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("\nTotal training time:", end - start, "sec")
        return result

    return wrapper


class TrainerVTS_V04c2:
    def __init__(self, img_encoder, img_decoder, csi_encoder, msk_encoder,
                 lr, epochs, cuda,
                 train_loader, valid_loader, test_loader,
                 ):

        self.models = {'imgen': img_encoder,
                       'imgde': img_decoder,
                       'csien': csi_encoder,
                       'msken': msk_encoder}
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda:" + str(cuda) if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.alpha = 0.8
        self.beta = 1.2

        self.recon_lossfun = nn.MSELoss(reduction='sum')
        self.bbx_loss = generalized_box_iou_loss

        self.temp_loss = {}
        self.loss = {'t': MyLoss_T_BBX(loss_terms=['LOSS', 'KL', 'RECON', 'BBX'],
                                       pred_terms=['GT', 'PRED', 'GT_BBX', 'PRED_BBX', 'IND']),
                     's': MyLoss_S_BBX(loss_terms=['LOSS', 'MU', 'LOGVAR', 'BBX', 'IMG'],
                                       pred_terms=['GT', 'T_PRED', 'S_PRED', 'T_LATENT', 'S_LATENT',
                                                   'GT_BBX', 'T_BBX', 'S_BBX', 'IND']),
                     }
        self.inds = None

    def current_title(self):
        """
        Shows current title
        :return: a string including current training epochs
        """
        return f"Te{self.loss['t'].epochs[-1]}_Se{self.loss['s'].epochs[-1]}"

    def vae_loss(self, pred, gt, mu, logvar):
        recon_loss = self.recon_lossfun(pred, gt) / pred.shape[0]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss * self.beta
        return loss, kl_loss, recon_loss

    def kd_loss(self, mu_s, logvar_s, bbx_s, mu_t, logvar_t, bbx_t):
        mu_loss = self.recon_lossfun(mu_s, mu_t) / mu_s.shape[0]
        logvar_loss = self.recon_lossfun(logvar_s, logvar_t) / logvar_s.shape[0]
        bbx_loss = self.bbx_loss(bbx_s, bbx_t) / bbx_s.shape[0]
        loss = self.alpha * mu_loss + (1 - self.alpha) * logvar_loss + bbx_loss
        return loss, mu_loss, logvar_loss, bbx_loss

    def calculate_loss_t(self, y, b, i=None):
        """
        Calculates loss function for back propagation,
        :param y: y data (image)
        :param b: b data (bounding box)
        :param i: index of data
        :return: loss object
        """
        mask = torch.where(y > 0, 1., 0.)
        latent, z, mu, logvar = self.models['imgen'](y)
        output = self.models['imgde'](z)
        bbx = self.models['msken'](mask)
        loss, kl_loss, recon_loss = self.vae_loss(output, y, mu, logvar)
        bbx_loss = self.bbx_loss(bbx, b)
        loss += bbx_loss
        self.temp_loss = {'LOSS': loss,
                          'KL': kl_loss,
                          'RECON': recon_loss,
                          'BBX': bbx_loss}
        return {'GT': y,
                'PRED': output,
                'GT_BBX': b,
                'BBX': bbx,
                'IND': i}

    def calculate_loss_s(self, x, y, b, i=None):
        """
        Calculates loss function for back propagation.
        :param x: x data (CSI)
        :param y: y data (image)
        :param b:: b data (bounding box)
        :param i: index of data
        :return: loss object
        """
        s_latent, s_z, s_mu, s_logvar, s_bbx = self.models['csien'](x)
        with torch.no_grad():
            mask = torch.where(y > 0, 1., 0.)
            t_latent, t_z, t_mu, t_logvar = self.models['imgen'](y)
            s_output = self.models['imgde'](s_z)
            t_output = self.models['imgde'](t_z)
            t_bbx = self.models['msken'](mask)
            image_loss = self.recon_lossfun(s_output, y)
        loss, mu_loss, logvar_loss, bbx_loss = self.kd_loss(s_mu, s_logvar, s_bbx, t_mu, t_logvar, t_bbx)

        self.temp_loss = {'LOSS': loss,
                          'MU': mu_loss,
                          'LOGVAR': logvar_loss,
                          'BBX': bbx_loss,
                          'IMG': image_loss}
        return {'GT': y,
                'GT_BBX': b,
                'T_LATENT': t_latent,
                'S_LATENT': s_latent,
                'T_PRED': t_output,
                'S_PRED': s_output,
                'T_BBX': t_bbx,
                'S_BBX': s_bbx,
                'IND': i}

    @timer
    def train_teacher(self, autosave=False, notion=''):
        """
        Trains the teacher.
        :param autosave: whether to save model parameters. Default is False
        :param notion: additional notes in save name
        :return: trained teacher
        """

        optimizer = self.optimizer([{'params': self.models['csien'].parameters()},
                                    {'params': self.models['cside'].parameters()},
                                    {'params': self.models['msken'].parameters()}], lr=self.lr)
        self.loss['teacher'].logger(self.lr, self.epochs)
        for epoch in range(self.epochs):
            # =====================train============================
            self.models['imgen'].train()
            self.models['msken'].train()
            self.models['imgde'].train()
            EPOCH_LOSS = {'LOSS': [],
                          'KL': [],
                          'RECON': [],
                          'BBX': []}
            for idx, (data_x, data_y, data_b, index) in enumerate(self.train_loader, 0):
                data_y = data_y.to(torch.float32).to(self.device)
                data_b = data_b.to(torch.float32).to(self.device)
                optimizer.zero_grad()

                PREDS = self.calculate_loss_t(data_y, data_b)
                self.temp_loss['LOSS'].backward()
                optimizer.step()
                for key in EPOCH_LOSS.keys():
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

                if idx % (len(self.train_loader) // 5) == 0:
                    print(f"\rTeacher: epoch={epoch}/{self.epochs}, batch={idx}/{len(self.train_loader)}, "
                          f"loss={self.temp_loss['LOSS'].item()}", end='')

            for key in EPOCH_LOSS.keys():
                EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
            self.loss['t'].update('train', EPOCH_LOSS)

            # =====================valid============================
            self.models['imgen'].eval()
            self.models['imgde'].eval()
            self.models['msken'].eval()
            EPOCH_LOSS = {'LOSS': [],
                          'KL': [],
                          'RECON': [],
                          'BBX': []}

            for idx, (data_x, data_y, data_b, index) in enumerate(self.valid_loader, 0):
                data_y = data_y.to(torch.float32).to(self.device)
                data_b = data_b.to(torch.float32).to(self.device)
                with torch.no_grad():
                    PREDS = self.calculate_loss_t(data_y, data_b)
                for key in EPOCH_LOSS.keys():
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())
            for key in EPOCH_LOSS.keys():
                EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
            self.loss['t'].update('valid', EPOCH_LOSS)

        if autosave:
            save_path = f'../saved/{notion}/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.models['imgen'].state_dict(),
                       f"{save_path}{notion}_{self.models['imgen']}_{self.current_title()}.pth")
            torch.save(self.models['imgde'].state_dict(),
                       f"{save_path}{notion}_{self.models['imgde']}_{self.current_title()}.pth")
            torch.save(self.models['msken'].state_dict(),
                       f"{save_path}{notion}_{self.models['msken']}_{self.current_title()}.pth")

    @timer
    def train_student(self, autosave=False, notion=''):
        """
        Trains the student.
        :param autosave: whether to save model parameters. Default is False
        :param notion: additional notes in save name
        :return: trained student
        """
        optimizer = self.optimizer([{'params': self.models['csien'].parameters()}], lr=self.lr)
        self.loss['s'].logger(self.lr, self.epochs)
        for epoch in range(self.epochs):

            # =====================train============================
            self.models['imgen'].eval()
            self.models['imgde'].eval()
            self.models['msken'].eval()
            self.models['csien'].train()

            EPOCH_LOSS = {'LOSS': [],
                          'MU': [],
                          'LOGVAR': [],
                          'BBX': [],
                          'IMG': []}

            for idx, (data_x, data_y, data_b, index) in enumerate(self.train_loader, 0):
                data_x = data_x.to(torch.float32).to(self.device)
                data_y = data_y.to(torch.float32).to(self.device)
                data_b = data_b.to(torch.float32).to(self.device)

                PREDS = self.calculate_loss_s(data_x, data_y, data_b)
                optimizer.zero_grad()
                self.temp_loss['LOSS'].backward()
                optimizer.step()

                for key in EPOCH_LOSS.keys():
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

                if idx % (len(self.train_loader) // 5) == 0:
                    print(f"\rStudent: epoch={epoch}/{self.epochs}, batch={idx}/{len(self.train_loader)}, "
                          f"loss={self.temp_loss['LOSS'].item()}", end='')

            for key in EPOCH_LOSS.keys():
                EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
            self.loss['s'].update('train', EPOCH_LOSS)

            # =====================valid============================
            self.models['imgen'].eval()
            self.models['imgde'].eval()
            self.models['msken'].eval()
            self.models['csien'].eval()

            EPOCH_LOSS = {'LOSS': [],
                          'MU': [],
                          'LOGVAR': [],
                          'BBX': [],
                          'IMG': []}

            for idx, (data_x, data_y, data_b, index) in enumerate(self.valid_loader, 0):
                data_x = data_x.to(torch.float32).to(self.device)
                data_y = data_y.to(torch.float32).to(self.device)
                data_b = data_b.to(torch.float32).to(self.device)
                with torch.no_grad():
                    PREDS = self.calculate_loss_s(data_x, data_y, data_b)

                for key in EPOCH_LOSS.keys():
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

            for key in EPOCH_LOSS.keys():
                EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
            self.loss['s'].update('valid', EPOCH_LOSS)

        if autosave:
            save_path = f'../saved/{notion}/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.models['csien'].state_dict(),
                       f"{save_path}{notion}_{self.models['csien']}_{self.current_title()}.pth")

    def test_teacher(self, mode='test'):
        self.models['imgen'].eval()
        self.models['imgde'].eval()
        self.models['msken'].eval()

        EPOCH_LOSS = {'LOSS': [],
                      'KL': [],
                      'RECON': [],
                      'BBX': []}

        if mode == 'test':
            loader = self.test_loader
        elif mode == 'train':
            loader = self.train_loader

        for idx, (data_x, data_y, data_b, index) in enumerate(loader, 0):
            data_y = data_y.to(torch.float32).to(self.device)
            data_b = data_b.to(torch.float32).to(self.device)
            with torch.no_grad():
                for sample in range(loader.batch_size):
                    ind = index[sample][np.newaxis, ...]
                    img = data_y[sample][np.newaxis, ...]
                    bbx = data_b[sample][np.newaxis, ...]
                    PREDS = self.calculate_loss_t(y=img, b=bbx, i=ind)

                    for key in EPOCH_LOSS.keys():
                        EPOCH_LOSS[key].append(self.temp_loss[key].item())

                    self.loss['t'].update('pred', PREDS)

            if idx % (len(loader)//5) == 0:
                print(f"\rTeacher: test={idx}/{len(loader)}, loss={self.temp_loss['LOSS'].item()}", end='')

        self.loss['t'].update('test', EPOCH_LOSS)
        for key in EPOCH_LOSS.keys():
            EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
        print(f"\nTest finished. Average loss={EPOCH_LOSS}")

    def test_student(self, mode='test'):
        self.models['imgen'].eval()
        self.models['imgde'].eval()
        self.models['msken'].eval()
        self.models['csien'].eval()

        EPOCH_LOSS = {'LOSS': [],
                      'MU': [],
                      'LOGVAR': [],
                      'BBX': [],
                      'IMG': []}

        if mode == 'test':
            loader = self.test_loader
        elif mode == 'train':
            loader = self.train_loader

        for idx, (data_x, data_y, data_b, index) in enumerate(loader, 0):
            data_x = data_x.to(torch.float32).to(self.device)
            data_y = data_y.to(torch.float32).to(self.device)
            data_b = data_b.to(torch.float32).to(self.device)
            with torch.no_grad():
                for sample in range(loader.batch_size):
                    ind = index[sample][np.newaxis, ...]
                    csi = data_x[sample][np.newaxis, ...]
                    img = data_y[sample][np.newaxis, ...]
                    bbx = data_b[sample][np.newaxis, ...]
                    PREDS = self.calculate_loss_s(x=csi, y=img, b=bbx, i=ind)

                    for key in EPOCH_LOSS.keys():
                        EPOCH_LOSS[key].append(self.temp_loss[key].item())

                    self.loss['s'].update('pred', PREDS)

            if idx % (len(loader) // 5) == 0:
                print(f"\rStudent: test={idx}/{len(loader)}, loss={self.temp_loss['LOSS'].item()}", end='')

        self.loss['s'].update('test', EPOCH_LOSS)
        for key in EPOCH_LOSS.keys():
            EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
        print(f"\nTest finished. Average loss={EPOCH_LOSS}")

    def plot_train_loss(self, mode='t', double_y=False, autosave=False, notion=''):
        title = {'t': f"Teacher Training Status",
                 's': f"Student Training Status"}
        filename = {'t': f"{notion}_T_train_{self.current_title()}.jpg",
                    's': f"{notion}_S_train_{self.current_title()}.jpg"}

        save_path = f'../saved/{notion}/'
        filename = f"../saved/{notion}/{filename}"

        self.loss[mode].plot_train(title[mode], filename[mode], double_y, autosave, notion)

        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(filename)
        plt.show()

    def generate_indices(self, source, select_num):
        inds = np.random.choice(list(range(len(source))), select_num, replace=False)
        inds = np.sort(inds)
        self.inds = inds
        return inds

    def plot_test_t(self, select_ind=None, select_num=8, autosave=False, notion=''):
        title = {'PRED': "Teacher Test Predicts",
                 'BBX': "Teacher Test BBX Predicts",
                 'LOSS': "Teacher Test Loss"}
        filename = {'PRED': f"{notion}_T_predict_{self.current_title()}.jpg",
                    'BBX': f"{notion}_T_bbx_{self.current_title()}.jpg",
                    'LOSS': f"{notion}_T_test_{self.current_title()}.jpg"}

        save_path = f'../saved/{notion}/'
        filename = f"../saved/{notion}/{filename}"
        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        if select_ind:
            inds = select_ind
        else:
            if self.inds:
                inds = self.inds
            else:
                inds = self.generate_indices(self.loss['t']['pred']['IND'], select_num)

        self.loss['t'].plot_predict(title['PRED'], inds, ('GT', 'PRED'))
        if autosave:
            plt.savefig(filename['PRED'])

        self.loss['t'].plot_bbx(title['BBX'], inds)
        if autosave:
            plt.savefig(filename['BBX'])

        self.loss['t'].plot_test(title['BBX'], inds)
        if autosave:
            plt.savefig(filename['LOSS'])

    def plot_test_s(self, select_ind=None, select_num=8, autosave=False, notion=''):
        title = {'PRED': "Student Test Predicts",
                 'BBX': "Student Test BBX Predicts",
                 'LOSS': "Student Test Loss",
                 'LATENT': f"Student Test Latents"}
        filename = {'PRED': f"{notion}_S_predict_{self.current_title()}.jpg",
                    'BBX': f"{notion}_S_bbx_{self.current_title()}.jpg",
                    'LOSS': f"{notion}_S_test_{self.current_title()}.jpg",
                    'LATENT': f"{notion}_S_latent_{self.current_title()}.jpg"}

        save_path = f'../saved/{notion}/'
        filename = f"../saved/{notion}/{filename}"
        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        if select_ind:
            inds = select_ind
        else:
            if self.inds:
                inds = self.inds
            else:
                inds = self.generate_indices(self.loss['s']['pred']['IND'], select_num)

        self.loss['s'].plot_predict(title['PRED'], inds, ('GT', 'T_PRED', 'S_PRED'))
        if autosave:
            plt.savefig(filename['PRED'])

        self.loss['s'].plot_bbx(title['BBX'], inds)
        if autosave:
            plt.savefig(filename['BBX'])

        self.loss['s'].plot_latent(title['s']['LATENT'], inds)
        if autosave:
            plt.savefig(filename['LATENT'])

        self.loss['t'].plot_test(title['BBX'], inds)
        if autosave:
            plt.savefig(filename['LOSS'])


