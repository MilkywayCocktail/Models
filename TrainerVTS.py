import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from TrainerTS import timer, MyDataset, split_loader, MyArgs, TrainerTeacherStudent

# ------------------------------------- #
# Trainer of VAE Teacher-student network


class TrainerVTS(TrainerTeacherStudent):
    def __init__(self, img_encoder, img_decoder, csi_encoder,
                 teacher_args, student_args,
                 train_loader, valid_loader, test_loader,
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
                                         div_loss=div_loss,
                                         img_loss=img_loss,
                                         temperature=temperature,
                                         alpha=alpha)
        self.batch_size = batch_size
        self.kl_weight = kl_weight

        self.plot_terms = {
            't_train': {'Total Loss': ['train', 'valid'],
                        'KL Loss': ['train_kl', 'valid_kl'],
                        'Recon Loss': ['train_recon', 'valid_recon']
                        },
            't_predict': {'Ground Truth': 'groundtruth',
                          'Estimated': 'predicts'
                          },
            't_test': {'Loss': 'loss',
                       'KL Loss': 'kl',
                       'Recon Loss': 'recon'
                       }
        }

    @staticmethod
    def __gen_teacher_train__():
        t_train_loss = {'learning_rate': [],
                        'epochs': [],
                        'train': [],
                        'valid': [],
                        'train_kl': [],
                        'valid_kl': [],
                        'train_recon': [],
                        'valid_recon': [],
                        }

        return t_train_loss

    @staticmethod
    def __gen_teacher_test__():
        t_test_loss = {'loss': [],
                       'recon': [],
                       'kl': [],
                       'predicts': [],
                       'groundtruth': []}
        return t_test_loss

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
        teacher_optimizer = self.args['t'].optimizer([{'params': self.img_encoder.parameters()},
                                                      {'params': self.img_decoder.parameters()}],
                                                     lr=self.args['t'].learning_rate)

        for epoch in range(self.args['t'].epochs):

            # =====================train============================
            self.img_encoder.train()
            self.img_decoder.train()
            train_epoch_loss = []
            kl_epoch_loss = []
            recon_epoch_loss = []
            for idx, (data_x, data_y) in enumerate(self.train_loader, 0):
                data_y = data_y.to(torch.float32).to(self.args['t'].device)
                teacher_optimizer.zero_grad()
                latent, z, mu, logvar = self.img_encoder(data_y)
                output = self.img_decoder(z)

                loss, kl_loss, recon_loss = self.loss(output, data_y, mu, logvar)

                loss.backward()
                teacher_optimizer.step()
                train_epoch_loss.append(loss.item())
                kl_epoch_loss.append(kl_loss.item())
                recon_epoch_loss.append(recon_loss.item())

                if idx % (len(self.train_loader) // 5) == 0:
                    print("\rTeacher: epoch={}/{}, {}/{} of train, loss={}".format(
                        epoch, self.args['t'].epochs, idx, len(self.train_loader), loss.item()), end='')
            self.train_loss['t']['train'].append(np.average(train_epoch_loss))
            self.train_loss['t']['train_kl'].append(np.average(kl_epoch_loss))
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
                    latent, z, mu, logvar = self.img_encoder(data_y)
                    output = self.img_decoder(z)
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
                       f"../saved/{self.img_encoder}{self.current_title()}_{notion}.pth")
            torch.save(self.img_decoder.state_dict(),
                       f"../saved/{self.img_decoder}{self.current_title()}_{notion}.pth")

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
                latent, z, mu, logvar = self.img_encoder(data_y)
                output = self.img_decoder(z)
                loss, kl_loss, recon_loss = self.loss(output, data_y, mu, logvar)

            self.test_loss['t']['loss'].append(loss.item())
            self.test_loss['t']['kl'].append(kl_loss.item())
            self.test_loss['t']['recon'].append(recon_loss.item())
            self.test_loss['t']['predicts'].append(output.cpu().detach().numpy().squeeze())
            self.test_loss['t']['groundtruth'].append(data_y.cpu().detach().numpy().squeeze())

            if idx % (len(loader)//5) == 0:
                print("\rTeacher: {}/{} of test, loss={}".format(idx, len(loader), loss.item()), end='')

    def train_student(self, autosave=False, notion=''):
        self.logger(mode='s')
        student_optimizer = self.args['s'].optimizer(self.csi_encoder.parameters(),
                                                     lr=self.args['s'].learning_rate)

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
                    teacher_preds, t_z, t_mu, t_logvar = self.img_encoder(data_y)
                    image_preds = self.img_decoder(student_preds)

                image_loss = self.img_loss(image_preds, data_y)
                student_loss = self.args['s'].criterion(student_preds, teacher_preds)

                # distil_loss = self.div_loss(nn.functional.softmax(student_preds / self.temperature, -1),
                #                             nn.functional.softmax(teacher_preds / self.temperature, -1))
                distil_loss = self.div_loss(student_preds, teacher_preds)

                loss = self.alpha * student_loss + (1 - self.alpha) * distil_loss

                student_optimizer.zero_grad()
                loss.backward()
                student_optimizer.step()

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
                       f"../saved/{self.csi_encoder}{self.current_title()}_{notion}.pth")

    def traverse_latent(self, img_ind, dataset, dim1=0, dim2=1, granularity=11, autosave=False, notion=''):
        self.__plot_settings__()

        self.img_encoder.eval()
        self.img_decoder.eval()

        if img_ind >= len(dataset):
            img_ind = np.random.randint(len(dataset))

        try:
            data_y, data_x = dataset[img_ind]
        except ValueError:
            data_y = dataset[img_ind]
        data_y = data_y[np.newaxis, ...].to(torch.float32).to(self.args['t'].device)

        latent, z, mu, logvar = self.img_encoder(data_y)
        z = z.squeeze()
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
                z[dim1], z[dim2] = xi, yi
                output = self.img_decoder(torch.from_numpy(e).to(self.args['t'].device))
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