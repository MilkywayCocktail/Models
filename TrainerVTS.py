import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
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
                 batch_size=64,
                 kl_weight=0.0025
                 ):
        super(TrainerVTS, self).__init__(img_encoder=img_encoder, img_decoder=img_decoder, csi_encoder=csi_encoder,
                                         teacher_args=teacher_args, student_args=student_args,
                                         train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
                                         optimizer=optimizer,
                                         div_loss=div_loss,
                                         img_loss=img_loss,
                                         temperature=temperature,
                                         alpha=alpha)
        self.batch_size = batch_size
        self.kl_weight = kl_weight

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

    def loss(self, y, gt, mu, logvar):
        recon_loss = self.args['t'].criterion(gt, y) / self.batch_size
        kl_loss = self.kl_loss(mu, logvar)
        loss = recon_loss + kl_loss * self.kl_weight
        return loss, kl_loss, recon_loss

    def current_title(self):
        return 'Te' + str(self.train_loss['t']['epochs'][-1][-1]) + '_Se' + str(self.train_loss['s']['epochs'][-1][-1])

    @timer
    def train_teacher(self, autosave=False, notion=''):
        self.logger(mode='t')

        for epoch in range(self.args['t'].epochs):

            # =====================train============================
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

                loss, kl_loss, recon_loss = self.loss(output, data_y, mu, logvar)

                loss.backward()
                self.teacher_optimizer.step()
                train_epoch_loss.append(loss.item())
                kl_epoch_loss.append(kl_loss.item())
                recon_epoch_loss.append(recon_loss.item())

                if idx % (len(self.train_loader) // 2) == 0:
                    print("\rTeacher: epoch={}/{}, {}/{} of train, loss={}".format(
                        epoch, self.args['t'].epochs, idx, len(self.train_loader), loss.item()), end='')
            self.train_loss['t']['train'].append(np.average(train_epoch_loss))
            self.train_loss['t']['train_kls'].append(np.average(kl_epoch_loss))
            self.train_loss['t']['train_recon'].append(np.average(recon_epoch_loss))

            # =====================valid============================
            self.img_encoder.eval()
            self.img_decoder.eval()
            valid_epoch_loss = []
            valid_kl_epoch_loss = []
            valid_recon_epoch_loss = []

            for idx, (data_x, data_y) in enumerate(self.valid_loader, 0):
                data_y = data_y.to(torch.float32).to(self.args['t'].device)
                with torch.no_grad():
                    latent, mu, logvar = self.img_encoder(data_y)
                    output = self.img_decoder(latent)
                    recon_loss = self.args['t'].criterion(output, data_y)
                    kl_loss = self.kl_loss(mu, logvar)
                    loss = recon_loss + kl_loss

                valid_epoch_loss.append(loss.item())
                valid_kl_epoch_loss.append(kl_loss.item())
                valid_recon_epoch_loss.append(recon_loss.item())
            self.train_loss['t']['valid'].append(np.average(valid_epoch_loss))
            self.train_loss['t']['valid_kl'].append(np.average(valid_kl_epoch_loss))
            self.train_loss['t']['valid_recon'].append(np.average(valid_recon_epoch_loss))

        if autosave:
            torch.save(self.img_encoder.state_dict(),
                       f"../Models/{self.img_encoder}{self.current_title()}_{notion}.pth")
            torch.save(self.img_decoder.state_dict(),
                       f"../Models/{self.img_decoder}{self.current_title()}_{notion}.pth")

    def test_teacher(self, mode='test'):
        self.test_loss['t'] = self.__gen_teacher_test__()
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
            with torch.no_grad():
                latent, mu, logvar = self.img_encoder(data_y)
                output = self.img_decoder(latent)
                loss, kl_loss, recon_loss = self.loss(output, data_y, mu, logvar)

            self.test_loss['t']['loss'].append(loss.item())
            self.test_loss['t']['kl'].append(kl_loss.item())
            self.test_loss['t']['recon'].append(recon_loss.item())
            self.test_loss['t']['predicts'].append(output.cpu().detach().numpy().squeeze().tolist())
            self.test_loss['t']['groundtruth'].append(data_y.cpu().detach().numpy().squeeze().tolist())

            if idx % (len(self.test_loader)//5) == 0:
                print("\rTeacher: {}/{} of test, loss={}".format(idx, len(loader), loss.item()), end='')

    def plot_teacher_loss(self, autosave=False, notion=''):
        self.__plot_settings__()

        loss_items = {'Total Loss': ['train_epochs', 'valid_epochs'],
                      'KL Loss': ['train_kl_epochs', 'valid_kl_epochs'],
                      'Recon Loss': ['train_recon_epochs', 'valid_recon_epochs']
                      }
        stage_color = self.colors(self.train_loss['t']['learning_rate'])
        line_color = ['b', 'orange']

        # Training & Validation Loss
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Teacher Training Status @ep{self.train_loss['t']['epochs'][-1]}")
        axes = fig.subplots(1, 3)
        axes = axes.flatten()

        for i, loss in enumerate(loss_items.keys()):
            for j, learning_rate in enumerate(self.train_loss['t']['learning_rate']):
                axes[i].axvline(self.train_loss['t']['epochs'][j],
                                linestyle='--',
                                color=stage_color[j],
                                label=f'lr={learning_rate}')

            axes[i].plot(self.train_loss['t'][loss_items[loss][1]], line_color[1], label=loss_items[loss][1])
            axes[i].plot(self.train_loss['t'][loss_items[loss][0]], line_color[0], label=loss_items[loss][0])
            axes[i].set_title(loss)
            axes[i].set_xlabel('#Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].grid()
            axes[i].legend()

        if autosave:
            plt.savefig(f"{self.current_title()}_T_train_{notion}.jpg")
        plt.show()

    def plot_teacher_test(self, select_num=8, autosave=False, notion=''):
        self.__plot_settings__()
        predict_items = {'Ground Truth': 'groundtruth',
                         'Estimated': 'predicts'
                         }

        # Depth Images
        inds = np.random.choice(list(range(len(self.test_loss['t']['groundtruth']))), select_num, replace=False)
        inds = np.sort(inds)
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Teacher Test Predicts @ep{self.train_loss['t']['epochs'][-1]}")
        subfigs = fig.subfigures(nrows=2, ncols=1)

        for i, item in enumerate(predict_items.keys()):
            subfigs[i].suptitle(predict_items[item])
            axes = subfigs[i].subplots(nrows=1, ncols=select_num)
            for j in range(len(axes)):
                img = axes[j].imshow(self.test_loss['t'][predict_items[item]][inds[j]])
                axes[j].axis('off')
                axes[j].set_title(f"#{inds[j]}")
            subfigs[i].colorbar(img, ax=axes, shrink=0.8)

        if autosave:
            plt.savefig(f"{self.current_title()}_T_predict_{notion}.jpg")
        plt.show()

        # Test Loss
        loss_items = {'Loss': 'loss',
                      'KL Loss': 'kl',
                      'Recon Loss': 'recon'
                      }
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Teacher Test Loss @ep{self.train_loss['t']['epochs'][-1]}")
        axes = fig.subplots(nrows=1, ncols=3)

        for i, loss in enumerate(loss_items.keys()):
            axes[i].scatter(list(range(len(self.test_loss['t']['groundtruth']))),
                            self.test_loss['t'][loss_items[loss]], alpha=0.6)
            axes[i].set_title(loss)
            axes[i].set_xlabel('#Sample')
            axes[i].set_ylabel('Loss')
            axes[i].grid()
            for j in inds:
                axes[i].scatter(j, self.test_loss['t'][loss_items[loss]][j], c='magenta', marker=(5, 1), linewidths=4)

        if autosave:
            plt.savefig(f"{self.current_title()}_T_test_{notion}.jpg")
        plt.show()

    def train_student(self, autosave=False, notion=''):
        self.logger(mode='s')

        for epoch in range(self.args['s'].epochs):

            # =====================train============================
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

                if idx % (len(self.train_loader) // 5) == 0:
                    print("\rStudent: epoch={}/{},{}/{}of train, student loss={}, distill loss={}".format(
                        epoch, self.args['s'].epochs, idx, len(self.train_loader),
                        loss.item(), distil_loss.item()), end='')

            self.train_loss['s']['train_epochs'].append(np.average(train_epoch_loss))
            self.train_loss['s']['train_straight'].append(np.average(straight_epoch_loss))
            self.train_loss['s']['train_distil'].append(np.average(distil_epoch_loss))
            self.train_loss['s']['train_image'].append(np.average(image_epoch_loss))

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
                with torch.no_grad():
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

            self.train_loss['s']['valid'].append(np.average(valid_epoch_loss))
            self.train_loss['s']['valid_straight'].append(np.average(straight_epoch_loss))
            self.train_loss['s']['valid_distil'].append(np.average(distil_epoch_loss))
            self.train_loss['s']['valid_image'].append(np.average(image_epoch_loss))

        if autosave:
            torch.save(self.csi_encoder.state_dict(),
                       f"../Models/{self.csi_encoder}{self.current_title()}_{notion}.pth")
