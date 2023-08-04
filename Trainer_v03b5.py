import numpy as np
import matplotlib.pyplot as plt
from Trainer_v03b3 import ImageEncoderM1, ImageDecoderM1
from Trainer_v03b2 import *
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
        terms = {'train': {'Total Loss': ['train', 'valid'],
                           'KL Loss': ['train_kl', 'valid_kl'],
                           'Recon Loss': ['train_recon', 'valid_recon'],
                           'Latent Loss': ['train_latent', 'valid_latent']
                           },
                 'predict': {'Ground Truth': 'groundtruth',
                             'Estimated': 'predicts',
                             'Re-Estimated': 're_predicts'
                             },
                 'test': {'Loss': 'loss',
                          'KL Loss': 'kl',
                          'Recon Loss': 'recon',
                          'Latent Loss': 'latent'
                          }
                 }
        return terms

    @staticmethod
    def __gen_teacher_train__():
        t_train_loss = {'learning_rate': [],
                        'epochs': [0],
                        'train': [],
                        'valid': [],
                        'train_kl': [],
                        'valid_kl': [],
                        'train_recon': [],
                        'valid_recon': [],
                        'train_latent': [],
                        'valid_latent': []
                        }

        return t_train_loss

    @staticmethod
    def __gen_teacher_test__():
        t_test_loss = {'loss': [],
                       'recon': [],
                       'kl': [],
                       'latent': [],
                       'predicts': [],
                       're_predicts': [],
                       'groundtruth': [],
                       'indices': []}
        return t_test_loss

    def loss(self, y, gt, latent, latent_p):
        # set reduce = 'sum'
        # considering batch
        recon_loss = self.args['t'].criterion(y, gt) / y.shape[0]
        latent_loss = self.args['s'].criterion(latent_p, latent) / y.shape[0]
        kl_loss = self.kl_loss(latent)
        loss = recon_loss + kl_loss * self.kl_weight + latent_loss
        return loss, kl_loss, recon_loss, latent_loss

    @timer
    def train_teacher(self, autosave=False, notion='', ret=''):
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
            latent_epoch_loss = []
            for idx, (data_x, data_y, index) in enumerate(self.train_loader, 0):
                data_y = data_y.to(torch.float32).to(self.args['t'].device)
                teacher_optimizer.zero_grad()
                latent, z = self.img_encoder(data_y)
                output = self.img_decoder(z)
                with torch.no_grad():
                    latent_p, z_p = self.img_encoder(output)

                loss, kl_loss, recon_loss, latent_loss = self.loss(output, data_y, latent, latent_p)

                loss.backward()
                teacher_optimizer.step()
                train_epoch_loss.append(loss.item())
                kl_epoch_loss.append(kl_loss.item())
                recon_epoch_loss.append(recon_loss.item())
                latent_epoch_loss.append(latent_loss.item())

                if idx % (len(self.train_loader) // 5) == 0:
                    print("\rTeacher: epoch={}/{}, {}/{} of train, recon_loss={}, kl_loss={}".format(
                        epoch, self.args['t'].epochs, idx, len(self.train_loader), loss.item(), kl_loss.item()),
                        end=ret)
            self.train_loss['t']['train'].append(np.average(train_epoch_loss))
            self.train_loss['t']['train_kl'].append(np.average(kl_epoch_loss))
            self.train_loss['t']['train_recon'].append(np.average(recon_epoch_loss))
            self.train_loss['t']['train_latent'].append(np.average(latent_epoch_loss))

            # =====================valid============================
            self.img_encoder.eval()
            self.img_decoder.eval()
            valid_epoch_loss = []
            valid_kl_epoch_loss = []
            valid_recon_epoch_loss = []
            valid_latent_epoch_loss = []

            for idx, (data_x, data_y, index) in enumerate(self.valid_loader, 0):
                data_y = data_y.to(torch.float32).to(self.args['t'].device)
                with torch.no_grad():
                    latent, z = self.img_encoder(data_y)
                    output = self.img_decoder(z)
                    latent_p, z_p = self.img_encoder(output)
                    loss, kl_loss, recon_loss, latent_loss = self.loss(output, data_y, latent, latent_p)

                valid_epoch_loss.append(loss.item())
                valid_kl_epoch_loss.append(kl_loss.item())
                valid_recon_epoch_loss.append(recon_loss.item())
                valid_latent_epoch_loss.append(latent_loss.item())
            self.train_loss['t']['valid'].append(np.average(valid_epoch_loss))
            self.train_loss['t']['valid_kl'].append(np.average(valid_kl_epoch_loss))
            self.train_loss['t']['valid_recon'].append(np.average(valid_recon_epoch_loss))
            self.train_loss['t']['valid_latent'].append(np.average(valid_recon_epoch_loss))

        if autosave:
            torch.save(self.img_encoder.state_dict(),
                       f"../saved/{self.img_encoder}{self.current_title()}_{notion}.pth")
            torch.save(self.img_decoder.state_dict(),
                       f"../saved/{self.img_decoder}{self.current_title()}_{notion}.pth")

    def test_teacher(self, mode='test'):
        self.test_loss['t'] = self.__gen_teacher_test__()
        self.img_encoder.eval()
        self.img_decoder.eval()
        test_epoch_loss = []
        test_kl_epoch_loss = []
        test_recon_epoch_loss = []
        test_latent_epoch_loss = []

        if mode == 'test':
            loader = self.test_loader
        elif mode == 'train':
            loader = self.train_loader

        for idx, (data_x, data_y, index) in enumerate(loader, 0):
            data_y = data_y.to(torch.float32).to(self.args['t'].device)
            if loader.batch_size == 1:
                data_y = data_y[np.newaxis, ...]
            with torch.no_grad():
                for sample in range(loader.batch_size):
                    image = data_y[sample][np.newaxis, ...]
                    latent, z = self.img_encoder(image)
                    output = self.img_decoder(z)
                    latent_p, z_p = self.img_encoder(output)
                    re_output = self.img_decoder(z_p)
                    loss, kl_loss, recon_loss, latent_loss = self.loss(output, image, latent, latent_p)

                    test_epoch_loss.append(loss.item())
                    test_kl_epoch_loss.append(kl_loss.item())
                    test_recon_epoch_loss.append(recon_loss.item())
                    test_latent_epoch_loss.append(latent_loss.item())
                    self.test_loss['t']['loss'].append(loss.item())
                    self.test_loss['t']['kl'].append(kl_loss.item())
                    self.test_loss['t']['recon'].append(recon_loss.item())
                    self.test_loss['t']['latent'].append(latent_loss.item())
                    self.test_loss['t']['predicts'].append(output.cpu().detach().numpy().squeeze())
                    self.test_loss['t']['re_predicts'].append(re_output.cpu().detach().numpy().squeeze())
                    self.test_loss['t']['groundtruth'].append(data_y.cpu().detach().numpy().squeeze())
                    self.test_loss['t']['indices'].append(index.cpu().detach().numpy().squeeze())

            if idx % (len(loader)//5) == 0:
                print("\rTeacher: {}/{} of test, loss={}".format(idx, len(loader), loss.item()), end='')

        avg_loss = np.mean(test_epoch_loss)
        avg_kl_loss = np.mean(test_kl_epoch_loss)
        avg_recon_loss = np.mean(test_recon_epoch_loss)
        avg_latent_loss = np.mean(test_latent_epoch_loss)
        print(f"\nTest finished. Average loss: total={avg_loss}, kl={avg_kl_loss}, recon={avg_recon_loss}, "
              f"latent={avg_latent_loss}")

    def plot_teacher_test(self, select_num=8, autosave=False, notion=''):
        self.__plot_settings__()
        predict_items = self.plot_terms['t']['predict']

        # Depth Images
        inds = np.random.choice(list(range(len(self.test_loss['t']['indices']))), select_num)
        samples = np.array(self.test_loss['t']['indices'])[inds]

        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Teacher Test Predicts @ep{self.train_loss['t']['epochs'][-1]}")
        subfigs = fig.subfigures(nrows=3, ncols=1)

        for i, item in enumerate(predict_items.keys()):
            subfigs[i].suptitle(predict_items[item])
            axes = subfigs[i].subplots(nrows=1, ncols=select_num)
            for j in range(len(axes)):
                img = axes[j].imshow(self.test_loss['t'][predict_items[item]][inds[j]], vmin=0, vmax=1)
                axes[j].axis('off')
                axes[j].set_title(f"#{samples[j]}")
            subfigs[i].colorbar(img, ax=axes, shrink=0.8)

        if autosave:
            plt.savefig(f"{self.current_title()}_T_predict_{notion}.jpg")
        plt.show()

        # Test Loss
        loss_items = self.plot_terms['t']['test']
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Teacher Test Loss @ep{self.train_loss['t']['epochs'][-1]}")
        if len(loss_items.keys()) == 1:
            axes = [plt.gca()]
        elif len(loss_items.keys()) > 3:
            axes = fig.subplots(2, np.ceil(len(loss_items.keys())).astype(int))
            axes = axes.flatten()
        else:
            axes = fig.subplots(1, len(loss_items.keys()))
            axes = axes.flatten()

        for i, loss in enumerate(loss_items.keys()):

            axes[i].scatter(list(range(len(self.test_loss['t']['groundtruth']))),
                            self.test_loss['t'][loss_items[loss]], alpha=0.6)
            axes[i].set_title(loss)
            axes[i].set_xlabel('#Sample')
            axes[i].set_ylabel('Loss')
            axes[i].grid()
            for j in inds:
                axes[i].scatter(j, self.test_loss['t'][loss_items[loss]][j],
                                c='magenta', marker=(5, 1), linewidths=4)

        if autosave:
            plt.savefig(f"{self.current_title()}_T_test_{notion}.jpg")
        plt.show()
