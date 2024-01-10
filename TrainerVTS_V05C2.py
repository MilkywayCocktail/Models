import torch
import torch.nn as nn
from torchvision.ops import generalized_box_iou_loss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.ops import generalized_box_iou_loss
import os
import time
from Loss import MyLoss, MyLoss_S

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


class MyLoss_T(MyLoss):
    def __init__(self, loss_terms, pred_terms):
        super(MyLoss_T, self).__init__(loss_terms, pred_terms)

    def plot_latent(self, title, select_ind):
        self.__plot_settings__()

        title = f"{title} @ep{self.epochs[-1]}"
        samples = np.array(self.loss['pred']['IND'])[select_ind]

        fig = plt.figure(constrained_layout=True)
        fig.suptitle(title)
        axes = fig.subplots(nrows=2, ncols=np.ceil(len(select_ind) / 2).astype(int))
        axes = axes.flatten()
        for j in range(len(select_ind)):
            axes[j].bar(range(len(self.loss['pred']['LAT'][select_ind[0]])),
                        self.loss['pred']['LAT'][select_ind[j]],
                        width=1, fc='blue', alpha=0.8, label='T_Latent')
            axes[j].set_ylim(-1, 1)
            axes[j].set_title(f"#{samples[j]}")
            axes[j].grid()

        axes[0].legend()
        plt.show()
        return fig


class MyLoss_S_BBX(MyLoss_S):
    def __init__(self, loss_terms, pred_terms):
        super(MyLoss_S_BBX, self).__init__(loss_terms, pred_terms)

    def plot_bbx(self, title, select_ind):
        self.__plot_settings__()

        title = f"{title} @ep{self.epochs[-1]}"
        samples = np.array(self.loss['pred']['IND'])[select_ind]

        fig = plt.figure(constrained_layout=True)
        fig.suptitle(title)
        axes = fig.subplots(nrows=2, ncols=np.ceil(len(select_ind) / 2).astype(int))
        axes = axes.flatten()
        for j in range(len(select_ind)):
            axes[j].set_xlim([0, 226])
            axes[j].set_ylim([0, 128])
            x, y, w, h = self.loss['pred']['GT_BBX'][select_ind[j]]
            axes[j].add_patch(Rectangle((x, y), w, h, edgecolor='blue', fill=False, lw=4, label='GroundTruth'))
            x, y, w, h = self.loss['pred']['S_BBX'][select_ind[j]]
            axes[j].add_patch(Rectangle((x, y), w, h, edgecolor='orange', fill=False, lw=4, label='Student'))
            axes[j].axis('off')
            axes[j].set_title(f"#{samples[j]}")

        axes[0].legend()
        plt.show()
        return fig


class TrainerVTS_V05c2:
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
        self.loss = {'t': MyLoss_T(loss_terms=['LOSS', 'KL', 'RECON'],
                                   pred_terms=['GT', 'PRED', 'LAT', 'IND']),
                     's': MyLoss_S_BBX(loss_terms=['LOSS', 'MU', 'LOGVAR', 'BBX', 'IMG'],
                                   pred_terms=['GT', 'T_PRED', 'S_PRED', 'T_LATENT', 'S_LATENT',
                                               'GT_BBX', 'S_BBX', 'IND']),
                     }
        self.inds = None

    def current_title(self):
        """
        Shows current title
        :return: a string including current training epochs
        """
        return f"Te{self.loss['t'].epochs[-1]}_Se{self.loss['s'].epochs[-1]}"

    @staticmethod
    def bbx_loss(bbx1, bbx2):
        # x, y, w, h to x1, y1, x2, y2
        # bbx1[..., -1] = bbx1[..., -1] + bbx1[..., -3]
        # bbx1[..., -2] = bbx1[..., -2] + bbx1[..., -4]
        # bbx2[..., -1] = bbx2[..., -1] + bbx2[..., -3]
        # bbx2[..., -2] = bbx2[..., -2] + bbx2[..., -4]
        return generalized_box_iou_loss(bbx1, bbx2, reduction='sum')

    def vae_loss(self, pred, gt, mu, logvar):
        recon_loss = self.recon_lossfun(pred, gt) / pred.shape[0]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss * self.beta
        return loss, kl_loss, recon_loss

    def kd_loss(self, mu_s, logvar_s, mu_t, logvar_t):
        mu_loss = self.recon_lossfun(mu_s, mu_t) / mu_s.shape[0]
        logvar_loss = self.recon_lossfun(logvar_s, logvar_t) / logvar_s.shape[0]
        loss = self.alpha * mu_loss + (1 - self.alpha) * logvar_loss
        return loss, mu_loss, logvar_loss

    def calculate_loss_t(self, img, i=None):

        z, mu, logvar = self.models['imgen'](img)
        output = self.models['imgde'](z)
        loss, kl_loss, recon_loss = self.vae_loss(output, img, mu, logvar)

        self.temp_loss = {'LOSS': loss,
                          'KL': kl_loss,
                          'RECON': recon_loss
                          }
        return {'GT': img,
                'PRED': output,
                'LAT': torch.cat((mu, logvar), -1),
                'IND': i
                }

    def calculate_loss_s_latent(self, csi, c_img, i=None):

        s_z, s_mu, s_logvar, s_bbx = self.models['csien'](csi)

        with torch.no_grad():
            t_z, t_mu, t_logvar = self.models['imgen'](c_img)
            s_output = self.models['imgde'](s_z)
            t_output = self.models['imgde'](t_z)
            image_loss = self.recon_lossfun(s_output, c_img)

        loss_i, mu_loss_i, logvar_loss_i = self.kd_loss(s_mu, s_logvar, t_mu, t_logvar)
        loss = loss_i

        self.temp_loss = {'LOSS': loss,
                          'MU': mu_loss_i,
                          'LOGVAR': logvar_loss_i,
                          'IMG': image_loss}
        return {'GT': c_img,
                'T_LATENT': torch.cat((t_mu, t_logvar), -1),
                'S_LATENT': torch.cat((s_mu, s_logvar), -1),
                'T_PRED': t_output,
                'S_PRED': s_output,
                'IND': i}

    def calculate_loss_s_bbx(self, csi, c_img, bbx, i=None):

        s_z, s_mu, s_logvar, s_bbx = self.models['csien'](csi)
        bbx_loss = self.bbx_loss(s_bbx, bbx)

        loss = bbx_loss

        self.temp_loss = {'LOSS': loss,
                          'BBX': bbx_loss}
        return {'GT': c_img,
                'GT_BBX': bbx,
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

        optimizer = self.optimizer([{'params': self.models['imgen'].parameters()},
                                    {'params': self.models['imgde'].parameters()}], lr=self.lr)
        self.loss['t'].logger(self.lr, self.epochs)
        for epoch in range(self.epochs):
            # =====================train============================
            self.models['imgen'].train()
            self.models['imgde'].train()
            EPOCH_LOSS = {'LOSS': [],
                          'KL': [],
                          'RECON': []
                          }
            for idx, (csi, img, bbx, index) in enumerate(self.train_loader, 0):
                img = img.to(torch.float32).to(self.device)
                optimizer.zero_grad()

                PREDS = self.calculate_loss_t(img)
                self.temp_loss['LOSS'].backward()
                optimizer.step()
                for key in EPOCH_LOSS.keys():
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

                if idx % (len(self.train_loader) // 5) == 0:
                    print(f"\rTeacher: epoch={epoch}/{self.epochs}, batch={idx}/{len(self.train_loader)}, "
                          f"loss={self.temp_loss['LOSS'].item():.4f}", end='')

            for key in EPOCH_LOSS.keys():
                EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
            self.loss['t'].update('train', EPOCH_LOSS)

            # =====================valid============================
            self.models['imgen'].eval()
            self.models['imgde'].eval()
            EPOCH_LOSS = {'LOSS': [],
                          'KL': [],
                          'RECON': []
                          }

            for idx, (csi, img, bbx, index) in enumerate(self.valid_loader, 0):
                img = img.to(torch.float32).to(self.device)
                with torch.no_grad():
                    PREDS = self.calculate_loss_t(img)
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

    @timer
    def train_student(self, autosave=False, notion='', mode='latent'):
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
            self.models['csien'].train()

            if mode == 'latent':
                EPOCH_LOSS = {'LOSS': [],
                              'MU': [],
                              'LOGVAR': [],
                              'IMG': []}
            elif mode == 'bbx':
                EPOCH_LOSS = {'LOSS': [],
                              'BBX': []}

            for idx, (csi, img, bbx, index) in enumerate(self.train_loader, 0):
                csi = csi.to(torch.float32).to(self.device)
                img = img.to(torch.float32).to(self.device)
                bbx = bbx.to(torch.float32).to(self.device)

                PREDS = self.calculate_loss_s(csi, img, bbx)
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
            self.models['csien'].eval()

            if mode == 'latent':
                EPOCH_LOSS = {'LOSS': [],
                              'MU': [],
                              'LOGVAR': [],
                              'IMG': []}
            elif mode == 'bbx':
                EPOCH_LOSS = {'LOSS': [],
                              'BBX': []}

            for idx, (csi, img, bbx, index) in enumerate(self.valid_loader, 0):
                csi = csi.to(torch.float32).to(self.device)
                img = img.to(torch.float32).to(self.device)
                bbx = bbx.to(torch.float32).to(self.device)
                with torch.no_grad():
                    PREDS = self.calculate_loss_s(csi, img, bbx)

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

        EPOCH_LOSS = {'LOSS': [],
                      'KL': [],
                      'RECON': []
                      }

        if mode == 'test':
            loader = self.test_loader
        elif mode == 'train':
            loader = self.train_loader

        self.loss['t'].reset('test')
        self.loss['t'].reset('pred')

        for idx, (csi, img, bbx, index) in enumerate(loader, 0):
            img = img.to(torch.float32).to(self.device)
            with torch.no_grad():
                for sample in range(loader.batch_size):
                    ind_ = index[sample][np.newaxis, ...]
                    img_ = img[sample][np.newaxis, ...]
                    PREDS = self.calculate_loss_t(img_, ind_)

                    for key in EPOCH_LOSS.keys():
                        EPOCH_LOSS[key].append(self.temp_loss[key].item())

                    self.loss['t'].update('pred', PREDS)

            if idx % (len(loader)//5) == 0:
                print(f"\rTeacher: test={idx}/{len(loader)}, loss={self.temp_loss['LOSS'].item():.4f}", end='')

        self.loss['t'].update('test', EPOCH_LOSS)
        for key in EPOCH_LOSS.keys():
            EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
        print(f"\nTest finished. Average loss={EPOCH_LOSS}")

    def test_student(self, mode='test'):
        self.models['imgen'].eval()
        self.models['imgde'].eval()
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
        self.loss['s'].reset('test')
        self.loss['s'].reset('pred')

        for idx, (csi, img, bbx, index) in enumerate(loader, 0):
            csi = csi.to(torch.float32).to(self.device)
            img = img.to(torch.float32).to(self.device)
            bbx = bbx.to(torch.float32).to(self.device)
            with torch.no_grad():
                for sample in range(loader.batch_size):
                    ind_ = index[sample][np.newaxis, ...]
                    csi_ = csi[sample][np.newaxis, ...]
                    img_ = img[sample][np.newaxis, ...]
                    bbx_ = bbx[sample][np.newaxis, ...]
                    PREDS = self.calculate_loss_s(csi_, img_, bbx_, ind_)

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

        fig = self.loss[mode].plot_train(title[mode], 'all', double_y)

        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig.savefig(f"{save_path}{filename[mode]}")
        plt.show()

    def generate_indices(self, source, select_num):
        inds = np.random.choice(list(range(len(source))), select_num, replace=False)
        inds = np.sort(inds)
        self.inds = inds
        return inds

    def plot_test_t(self, select_ind=None, select_num=8, autosave=False, notion=''):
        title = {'PRED': "Teacher Test IMG Predicts",
                 'LAT': "Teacher Latents",
                 'LOSS': "Teacher Test Loss"}
        filename = {'PRED': f"{notion}_T_img_{self.current_title()}.jpg",
                    "LAT": f"{notion}_T_latent_{self.current_title()}.jpg",
                    'LOSS': f"{notion}_T_test_{self.current_title()}.jpg"}

        save_path = f'../saved/{notion}/'

        if select_ind:
            inds = select_ind
        else:
            if self.inds is not None and select_num == len(self.inds):
                inds = self.inds
            else:
                inds = self.generate_indices(self.loss['t'].loss['pred']['IND'], select_num)

        fig1 = self.loss['t'].plot_predict(title['PRED'], inds, ('GT', 'PRED'))
        fig2 = self.loss['t'].plot_latent(title['LAT'], inds)
        fig3 = self.loss['t'].plot_test(title['LOSS'], inds)

        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig1.savefig(f"{save_path}{filename['PRED']}")
            fig2.savefig(f"{save_path}{filename['LAT']}")
            fig3.savefig(f"{save_path}{filename['LOSS']}")

    def plot_test_s(self, select_ind=None, select_num=8, autosave=False, notion='', mode='latent'):
        title = {'PRED': "Student Test IMG Predicts",
                 'BBX': "Student Test BBX Predicts",
                 'LOSS': "Student Test Loss",
                 'LATENT': f"Student Test Latents for IMG"}
        filename = {'PRED': f"{notion}_S_img_{self.current_title()}.jpg",
                    'BBX': f"{notion}_S_bbx_{self.current_title()}.jpg",
                    'LOSS': f"{notion}_S_test_{self.current_title()}.jpg",
                    'LATENT': f"{notion}_S_latent_{self.current_title()}.jpg",}

        save_path = f'../saved/{notion}/'

        if select_ind:
            inds = select_ind
        else:
            if self.inds is not None and select_num == len(self.inds):
                inds = self.inds
            else:
                inds = self.generate_indices(self.loss['s'].loss['pred']['IND'], select_num)

        if mode == 'latent':
            fig1 = self.loss['s'].plot_predict(title['PRED'], inds, ('GT', 'T_PRED', 'S_PRED'))
            fig3 = self.loss['s'].plot_latent(title['LATENT'], inds, ('T_LATENT', 'S_LATENT'))
            fig4 = self.loss['s'].plot_test(title['LOSS'], inds)

            if autosave:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                fig1.savefig(f"{save_path}{filename['PRED']}")
                fig3.savefig(f"{save_path}{filename['LATENT']}")
                fig4.savefig(f"{save_path}{filename['LOSS']}")

        elif mode == 'bbx':
            fig2 = self.loss['s'].plot_bbx(title['BBX'], inds)

            if autosave:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                fig2.savefig(f"{save_path}{filename['BBX']}")

    def scheduler(self, train_t=True, train_s=True,
                  t_turns=10, s_turns=10,
                  lr_decay=False, decay_rate=0.4,
                  test_mode='train', select_num=8,
                  autosave=False, notion=''):
        """
        Schedules the process of training and testing.
        :param train_t: whether to train the teacher. True or False. Default is True
        :param train_s: whether to train the student. True or False. Default is True
        :param t_turns: number of turns to run teacher train-test operations. Default is 10
        :param s_turns: number of turns to run student train-test operations. Default is 10
        :param lr_decay: whether to decay learning rate in training. Default is False
        :param decay_rate: decay rate of learning rate. Default it 0.4
        :param test_mode: 'train' or 'test' (data loader). Default is 'train'
        :param select_num: Number of samples to show in results. Default is 8
        :param autosave: whether to save the plots. Default is False
        :param notion: additional notes in save name
        :return: trained models and test results
        """
        if train_t:
            for i in range(t_turns):
                self.train_teacher()
                self.test_teacher(mode=test_mode)
                self.plot_test_t(select_num=select_num, autosave=autosave, notion=notion)
                self.plot_train_loss(mode='t', autosave=autosave, notion=notion)
                if lr_decay:
                    self.lr *= decay_rate

        if train_s:
            for i in range(s_turns):
                self.train_student()
                self.test_student(mode=test_mode)
                self.plot_test_s(select_num=select_num, autosave=autosave, notion=notion)
                self.plot_train_loss(mode='s', autosave=autosave, notion=notion)
                if lr_decay:
                    self.lr *= decay_rate

        print("\nSchedule Completed!")
