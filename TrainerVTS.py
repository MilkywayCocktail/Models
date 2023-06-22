import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from TrainerTS import timer, MyDataset, split_loader, MyArgs, TrainerTeacherStudent

# ------------------------------------- #
# Trainer of VAE Teacher-student network


class TrainerVTS(TrainerTeacherStudent):
    def __init__(self, img_encoder, img_decoder, csi_encoder,
                 teacher_args, student_args,
                 train_loader, valid_loader, test_loader,
                 optimizer=torch.optim.Adam,
                 div_loss=nn.KLDivLoss(reduction='batchmean'),
                 img_loss=nn.SmoothL1Loss(),
                 temperature=20,
                 alpha=0.3,
                 latent_dim=8
                 ):
        super(TrainerVTS, self).__init__(img_encoder=img_encoder, img_decoder=img_decoder, csi_encoder=csi_encoder,
                                         teacher_args=teacher_args, student_args=student_args,
                                         train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
                                         optimizer=optimizer,
                                         div_loss=div_loss,
                                         img_loss=img_loss,
                                         temperature=temperature,
                                         alpha=alpha)
        self.latent_dim = latent_dim

    @staticmethod
    def __gen_teacher_train__():
        t_train_loss = {'learning_rate': [],
                        'epochs': [],
                        'train_epochs': [],
                        'valid_epochs': [],
                        'train_kl_epochs': [],
                        'valid_kl_epochs': [],
                        'train_recon_epochs': [],
                        'valid_recon_epochs': [],
                        }

        return t_train_loss

    @staticmethod
    def __gen_teacher_test__():
        test_loss = {'loss': [],
                     'recon': [],
                     'kl': [],
                     'predicts': [],
                     'groundtruth': []}
        return test_loss

    @staticmethod
    def kl_loss(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def current_title(self):
        return 'Te' + str(self.train_loss['t']['epochs'][-1][-1]) + '_Se' + str(self.train_loss['s']['epochs'][-1][-1])

    @timer
    def train_teacher(self, autosave=False, notion=''):
        self.logger(mode='t')

        for epoch in range(self.args['t'].epochs):
            self.img_encoder.train()
            self.img_decoder.train()
            train_epoch_loss = []
            kl_epoch_loss = []
            recon_epoch_loss = []
            for idx, (data_x, data_y) in enumerate(self.train_loader, 0):
                data_y = data_y.to(torch.float32).to(self.args['t'].device)
                self.teacher_optimizer.zero_grad()
                latent, mu, logvar = self.img_encoder(data_y)
                output = self.img_decoder(latent)

                recon_loss = self.args['t'].criterion(output, data_y)
                kl_loss = self.kl_loss(mu, logvar)
                loss = recon_loss + kl_loss

                loss.backward()
                self.teacher_optimizer.step()
                train_epoch_loss.append(loss.item())
                kl_epoch_loss.append(kl_loss.item())
                recon_epoch_loss.append(recon_loss.item())

                if idx % (len(self.train_loader) // 2) == 0:
                    print("\rTeacher: epoch={}/{},{}/{}of train, loss={}".format(
                        epoch, self.args['t'].epochs, idx, len(self.train_loader), loss.item()), end='')
            self.train_loss['t']['train_epochs'].append(np.average(train_epoch_loss))
            self.train_loss['t']['train_kl_epochs'].append(np.average(kl_epoch_loss))
            self.train_loss['t']['train_recon_epochs'].append(np.average(recon_epoch_loss))

        if autosave:
            torch.save(self.img_encoder.state_dict(),
                       f"../Models/{self.img_encoder}{self.current_title()}_{notion}.pth")
            torch.save(self.img_decoder.state_dict(),
                       f"../Models/{self.img_decoder}{self.current_title()}_{notion}.pth")

        # =====================valid============================
        self.img_encoder.eval()
        self.img_decoder.eval()
        valid_epoch_loss = []
        valid_kl_epoch_loss = []
        valid_recon_epoch_loss = []

        for idx, (data_x, data_y) in enumerate(self.valid_loader, 0):
            data_y = data_y.to(torch.float32).to(self.args['t'].device)
            latent, mu, logvar = self.img_encoder(data_y)
            output = self.img_decoder(latent)

            recon_loss = self.args['t'].criterion(output, data_y)
            kl_loss = self.kl_loss(mu, logvar)
            loss = recon_loss + kl_loss

            valid_epoch_loss.append(loss.item())
            valid_kl_epoch_loss.append(kl_loss.item())
            valid_recon_epoch_loss.append(recon_loss.item())
        self.train_loss['t']['valid_epochs'].append(np.average(valid_epoch_loss))
        self.train_loss['t']['valid_kl_epochs'].append(np.average(valid_kl_epoch_loss))
        self.train_loss['t']['valid_recon_epochs'].append(np.average(valid_recon_epoch_loss))

    def test_teacher(self, mode='test'):
        self.t_test_loss = self.__gen_teacher_test__()
        self.img_encoder.eval()
        self.img_decoder.eval()

        if mode == 'test':
            loader = self.test_loader
        elif mode == 'train':
            loader = self.train_loader

        for idx, (data_x, data_y) in enumerate(loader, 0):
            data_y = data_y.to(torch.float32).to(self.args['t'].device)
            if loader.batch_size != 1:
                data_y = data_y[0][np.newaxis, ...]

            latent, mu, logvar = self.img_encoder(data_y)
            output = self.img_decoder(latent)

            recon_loss = self.args['t'].criterion(output, data_y)
            kl_loss = self.kl_loss(mu, logvar)
            loss = recon_loss + kl_loss

            self.t_test_loss['loss'].append(loss.item())
            self.t_test_loss['kl'].append(kl_loss.item())
            self.t_test_loss['recon'].append(recon_loss.item())
            self.t_test_loss['predicts'].append(output.cpu().detach().numpy().squeeze().tolist())
            self.t_test_loss['groundtruth'].append(data_y.cpu().detach().numpy().squeeze().tolist())

            if idx % (len(self.test_loader)//5) == 0:
                print("\rTeacher: {}/{}of test, loss={}".format(idx, len(loader), loss.item()), end='')

    def plot_teacher_loss(self, autosave=False, notion=''):
        self.__plot_settings__()

        loss_items = {'Train': 'train_epochs',
                      'Train KL Loss': 'train_kl_epochs',
                      'Train Recon Loss': 'train_recon_epochs',
                      'Valid': 'valid_epochs',
                      'Valid KL Loss': 'valid_kl_epochs',
                      'Valid Recon Loss': 'valid_recon_epochs'}
        color = self.colors(self.train_loss['t']['learning_rate'])
        linecolor = ['b', 'b', 'b', 'orange', 'orange', 'orange']

        # Training Loss
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Teacher Train Loss @ep{self.train_loss['t']['epochs'][-1]}")
        axes = fig.subplots(2, 3)
        axes = axes.flatten()

        for i, loss in enumerate(loss_items.keys()):
            for j, learning_rate in enumerate(self.train_loss['t']['learning_rate']):

                axes[i].axvline(self.train_loss['t']['epochs'][j],
                                linestyle='--',
                                color=color[j],
                                label=f'lr={learning_rate}')

            axes[i].plot(self.train_loss['t'][loss_items[loss]], linecolor[i])
            axes[i].set_title(loss)
            axes[i].legend()
            axes.set_xlabel('#epoch')
            axes.set_ylabel('loss')
            axes.grid()

        if autosave:
            plt.savefig(f"{self.current_title()}_T_train_{notion}.jpg")
        plt.show()

    def plot_teacher_test(self, select_num=8, autosave=False, notion=''):
        self.__plot_settings__()

        # Depth Images
        imgs = np.random.choice(list(range(len(self.t_test_loss['groundtruth']))), select_num, replace=False)
        imgs = np.sort(imgs)
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Teacher Test Predicts @ep{self.train_loss['t']['epochs'][-1]}")
        subfigs = fig.subfigures(nrows=2, ncols=1)

        subfigs[0].suptitle('Ground Truth')
        ax = subfigs[0].subplots(nrows=1, ncols=select_num)
        for a in range(len(ax)):
            ima = ax[a].imshow(self.t_test_loss['groundtruth'][imgs[a]])
            ax[a].axis('off')
            ax[a].set_title('#' + str(imgs[a]))
            ax[a].set_xlabel(str(imgs[a]))
        subfigs[0].colorbar(ima, ax=ax, shrink=0.8)

        subfigs[1].suptitle('Estimated')
        ax = subfigs[1].subplots(nrows=1, ncols=select_num)
        for a in range(len(ax)):
            imb = ax[a].imshow(self.t_test_loss['predicts'][imgs[a]])
            ax[a].axis('off')
            ax[a].set_title('#' + str(imgs[a]))
            ax[a].set_xlabel(str(imgs[a]))
        subfigs[1].colorbar(imb, ax=ax, shrink=0.8)

        if autosave:
            plt.savefig(f"{self.current_title()}_T_predict_{notion}.jpg")
        plt.show()

        # Test Loss
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Teacher Test Loss @ep{self.train_loss['t']['epochs'][-1]}")
        axes = fig.subplots(nrows=1, ncols=3)
        axes[0].set_title('Loss')
        axes[1].set_title('KL Loss')
        axes[2].set_title('Recon Loss')
        axes[0].scatter(list(range(len(self.t_test_loss['groundtruth']))), self.t_test_loss['loss'], alpha=0.6)
        axes[1].scatter(list(range(len(self.t_test_loss['groundtruth']))), self.t_test_loss['kl'], alpha=0.6)
        axes[2].scatter(list(range(len(self.t_test_loss['groundtruth']))), self.t_test_loss['recon'], alpha=0.6)
        for i in imgs:
            axes[0].scatter(i, self.t_test_loss['loss'][i], c='magenta', marker=(5, 1), linewidths=4)
            axes[1].scatter(i, self.t_test_loss['kl'][i], c='magenta', marker=(5, 1), linewidths=4)
            axes[2].scatter(i, self.t_test_loss['recon'][i], c='magenta', marker=(5, 1), linewidths=4)
        for ax in axes:
            ax.set_xlabel('#Sample')
            ax.set_ylabel('Loss')
            ax.grid()

        if autosave:
            plt.savefig(f"{self.current_title()}_T_test_{notion}.jpg")
        plt.show()

    def train_student(self, autosave=False, notion=''):

        for epoch in range(self.args['s'].epochs):
            self.img_encoder.eval()
            self.img_decoder.eval()
            self.csi_encoder.train()
            train_epoch_loss = []
            straight_epoch_loss = []
            distil_epoch_loss = []
            image_epoch_loss = []

            for idx, (data_x, data_y) in enumerate(self.train_loader, 0):
                data_x = data_x.to(torch.float32).to(self.args['s'].device)
                data_y = data_y.to(torch.float32).to(self.args['s'].device)

                student_preds, mu, logvar = self.csi_encoder(data_x)
                with torch.no_grad():
                    teacher_preds, t_mu, t_logvar = self.img_encoder(data_y)
                    image_preds = self.img_decoder(student_preds)

                image_loss = self.img_loss(image_preds, data_y)
                student_loss = self.args['s'].criterion(student_preds, teacher_preds)

                distil_loss = self.div_loss(nn.functional.softmax(student_preds / self.temperature, -1),
                                            nn.functional.softmax(teacher_preds / self.temperature, -1))

                loss = self.alpha * student_loss + (1 - self.alpha) * distil_loss

                self.student_optimizer.zero_grad()
                loss.backward()
                self.student_optimizer.step()

                train_epoch_loss.append(loss.item())
                straight_epoch_loss.append(student_loss.item())
                distil_epoch_loss.append(distil_loss.item())
                image_epoch_loss.append(image_loss.item())

                if idx % (len(self.train_loader) // 2) == 0:
                    print("\rStudent: epoch={}/{},{}/{}of train, student loss={}, distill loss={}".format(
                        epoch, self.args['s'].epochs, idx, len(self.train_loader),
                        loss.item(), distil_loss.item()), end='')

            self.train_loss['s']['train_epochs'].append(np.average(train_epoch_loss))
            self.train_loss['s']['train_straight_epochs'].append(np.average(straight_epoch_loss))
            self.train_loss['s']['train_distil_epochs'].append(np.average(distil_epoch_loss))
            self.train_loss['s']['train_image_epochs'].append(np.average(image_epoch_loss))

        if autosave:
            torch.save(self.csi_encoder.state_dict(),
                       f"../Models/{self.csi_encoder}{self.current_title()}_{notion}.pth")

        # =====================valid============================
        self.csi_encoder.eval()
        self.img_encoder.eval()
        self.img_decoder.eval()
        valid_epoch_loss = []
        straight_epoch_loss = []
        distil_epoch_loss = []
        image_epoch_loss = []

        for idx, (data_x, data_y) in enumerate(self.valid_loader, 0):
            data_x = data_x.to(torch.float32).to(self.args['s'].device)
            data_y = data_y.to(torch.float32).to(self.args['s'].device)

            teacher_preds = self.img_encoder(data_y)
            student_preds = self.csi_encoder(data_x)
            image_preds = self.img_decoder(student_preds)
            image_loss = self.img_loss(image_preds, data_y)

            student_loss = self.args['s'].criterion(student_preds, teacher_preds)

            distil_loss = self.div_loss(nn.functional.softmax(student_preds / self.temperature, -1),
                                        nn.functional.softmax(teacher_preds / self.temperature, -1))

            loss = self.alpha * student_loss + (1 - self.alpha) * distil_loss

            valid_epoch_loss.append(loss.item())
            straight_epoch_loss.append(student_loss.item())
            distil_epoch_loss.append(distil_loss.item())
            image_epoch_loss.append(image_loss.item())

        self.train_loss['s']['valid_epochs'].append(np.average(valid_epoch_loss))
        self.train_loss['s']['valid_straight_epochs'].append(np.average(straight_epoch_loss))
        self.train_loss['s']['valid_distil_epochs'].append(np.average(distil_epoch_loss))
        self.train_loss['s']['valid_image_epochs'].append(np.average(image_epoch_loss))


