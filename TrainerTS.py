import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import os

# ------------------------------------- #
# Trainer of Teacher-student network


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


def bn(channels, batchnorm):
    if batchnorm:
        return nn.BatchNorm2d(channels)
    else:
        return nn.Identity(channels)


class Interpolate(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class MyDataset(Data.Dataset):
    def __init__(self, x_path, y_path, img_size=(128, 128), transform=None, img='y', number=0):
        self.seeds = None
        self.img_size = img_size
        self.transform = transform
        self.img = img
        self.data = self.__load_data__(x_path, y_path, number=number)
        print('loaded')

    def __transform__(self, sample):
        if self.transform:
            return self.transform(Image.fromarray((np.array(sample)).squeeze(), mode='L'))
        else:
            return sample

    def __getitem__(self, index):
        if self.img == 'y':
            return self.data['x'][index], self.__transform__(self.data['y'][index]), index
        elif self.img == 'x':
            return self.__transform__(self.data['x'][index]), self.data['y'][index], index

    def __len__(self):
        return self.data['x'].shape[0]

    def __load_data__(self, x_path, y_path, number):
        x = np.load(x_path)
        y = np.load(y_path)
        if self.img == 'x':
            x = x.reshape((-1, 1, self.img_size[0], self.img_size[1]))
        elif self.img == 'y':
            y = y.reshape((-1, 1, self.img_size[0], self.img_size[1]))

        if x.shape[0] == y.shape[0]:
            total_count = x.shape[0]
            if number != 0:
                picked = np.random.choice(list(range(total_count)), size=number, replace=False)
                self.seeds = picked
                x = x[picked]
                y = y[picked]
        else:
            print(x.shape, y.shape, "lengths not equal!")

        return {'x': x, 'y': y}


class MnistDataset(MyDataset):
    def __init__(self, mnist, img_size=(28, 28), transform=None, swap_xy=False, number=0):
        MyDataset.__init__(x_path=None, y_path=None, img_size=img_size)
        self.seeds = None
        self.img_size = img_size
        self.transform = transform
        self.swap_xy = swap_xy
        self.data = self.__load_data__(mnist, number=number)
        print('loaded')

    def __load_data__(self, mnist, number):

        x = mnist[:, 0].reshape((-1, 1, self.img_size[0], self.img_size[1]))
        y = mnist[:, 1]

        if x.shape[0] == y.shape[0]:
            total_count = x.shape[0]
            if number != 0:
                picked = np.random.choice(list(range(total_count)), size=number, replace=False)
                self.seeds = picked
                x = x[picked]
                y = y[picked]
        else:
            print(x.shape, y.shape, "lengths not equal!")

        if self.swap_xy:
            return {'x': y, 'y': x}
        else:
            return {'x': x, 'y': y}


def split_loader(dataset, train_size, valid_size, test_size, batch_size):
    train_dataset, valid_dataset, test_dataset = Data.random_split(dataset, [train_size, valid_size, test_size])
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = Data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    return train_loader, valid_loader, test_loader


class MyArgs:
    def __init__(self, cuda=1, epochs=30, learning_rate=0.001,
                 criterion=nn.CrossEntropyLoss(),
                 optimizer=torch.optim.Adam):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda:" + str(cuda) if torch.cuda.is_available() else "cpu")
        self.criterion = criterion
        self.optimizer = optimizer


class TrainerTeacherStudent:

    def __init__(self, img_encoder, img_decoder, csi_encoder,
                 teacher_args, student_args,
                 train_loader, valid_loader, test_loader,
                 div_loss=nn.KLDivLoss(reduction='batchmean'),
                 img_loss=nn.SmoothL1Loss(),
                 temperature=20,
                 alpha=0.3,
                 latent_dim=8):
        self.img_encoder = img_encoder
        self.img_decoder = img_decoder
        self.csi_encoder = csi_encoder

        self.args = {'t': teacher_args,
                     's': student_args}

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.train_loss = {'t': self.__gen_teacher_train__(),
                           's': self.__gen_student_train__()}
        self.test_loss = {'t': self.__gen_teacher_test__(),
                          's': self.__gen_student_test__()}
        self.plot_terms = {'t': self.__teacher_plot_terms__(),
                           's': self.__student_plot_terms__()}

        self.div_loss = div_loss
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.temperature = temperature
        self.alpha = alpha
        self.img_loss = img_loss
        self.latent_dim = latent_dim

    @staticmethod
    def __plot_settings__():
        plt.rcParams['figure.figsize'] = (20, 10)
        plt.rcParams["figure.titlesize"] = 35
        plt.rcParams['lines.markersize'] = 10
        plt.rcParams['axes.titlesize'] = 30
        plt.rcParams['axes.labelsize'] = 30
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20

    @staticmethod
    def __gen_teacher_train__():
        t_train_loss = {'learning_rate': [],
                        'epochs': [0],
                        'train': [],
                        'valid': [],
                        }
        return t_train_loss

    @staticmethod
    def __gen_student_train__():
        s_train_loss = {'learning_rate': [],
                        'epochs': [0],
                        'train': [],
                        'valid': [],
                        'train_straight': [],
                        'valid_straight': [],
                        'train_distil': [],
                        'valid_distil': [],
                        'train_image': [],
                        'valid_image': []
                        }
        return s_train_loss

    @staticmethod
    def __gen_teacher_test__():
        t_test_loss = {'loss': [],
                       'predicts': [],
                       'groundtruth': []
                       }
        return t_test_loss

    @staticmethod
    def __gen_student_test__():
        s_test_loss = {'loss': [],
                       'latent_straight': [],
                       'latent_distil': [],
                       'image': [],
                       'predicts_t_latent': [],
                       'predicts_latent': [],
                       'predicts': [],
                       'groundtruth': []
                       }
        return s_test_loss

    @staticmethod
    def __teacher_plot_terms__():
        terms = {'train': {'Loss': ['train', 'valid']},
                 'predict': {'Ground Truth': 'groundtruth',
                             'Estimated': 'predicts'},
                 'test': {'Loss': 'loss'}
                 }
        return terms

    @staticmethod
    def __student_plot_terms__():
        terms = {'train': {'Student Loss': ['train', 'valid'],
                           'Straight Loss': ['train_straight', 'valid_straight'],
                           'Distillation Loss': ['train_distil', 'valid_distil'],
                           'Image Loss': ['train_image', 'valid_image']},
                 'predict': {'Ground Truth': 'groundtruth',
                             'Estimated': 'predicts'},
                 'test': {'Student Loss': 'loss',
                          'Straight Loss': 'latent_straight',
                          'Distillation Loss': 'latent_distil',
                          'Image Loss': 'image'}
                 }
        return terms

    def current_title(self):
        return f"Te{self.train_loss['t']['epochs'][-1]}_Se{self.train_loss['s']['epochs'][-1]}"

    @staticmethod
    def colors(arrays):
        arr = -np.log(arrays)
        norm = plt.Normalize(arr.min(), arr.max())
        map_vir = cm.get_cmap(name='viridis')
        c = map_vir(norm(arr))
        return c

    def logger(self, mode='t'):
        """
        Logs learning rate and number of epochs before training.
        :param mode: 't' or 's'
        :return:
        """

        # First round
        if not self.train_loss[mode]['learning_rate']:
            self.train_loss[mode]['learning_rate'].append(self.args[mode].learning_rate)
            self.train_loss[mode]['epochs'].append(self.args[mode].epochs)

        else:
            # Not changing learning rate
            if self.args[mode].learning_rate == self.train_loss[mode]['learning_rate'][-1]:
                self.train_loss[mode]['epochs'][-1] += self.args[mode].epochs

            # Changing learning rate
            if self.args[mode].learning_rate != self.train_loss[mode]['learning_rate'][-1]:
                last_end = self.train_loss[mode]['epochs'][-1]
                self.train_loss[mode]['learning_rate'].append(self.args[mode].learning_rate)
                self.train_loss[mode]['epochs'].append(last_end + self.args[mode].epochs)

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
            for idx, (data_x, data_y, index) in enumerate(self.train_loader, 0):
                data_y = data_y.to(torch.float32).to(self.args['t'].device)
                teacher_optimizer.zero_grad()
                latent = self.img_encoder(data_y).data
                output = self.img_decoder(latent)

                loss = self.args['t'].criterion(output, data_y)
                loss.backward()
                teacher_optimizer.step()
                train_epoch_loss.append(loss.item())

                if idx % (len(self.train_loader) // 5) == 0:
                    print("\rTeacher: epoch={}/{}, {}/{} of train, loss={}".format(
                        epoch, self.args['t'].epochs, idx, len(self.train_loader), loss.item()), end='')
            self.train_loss['t']['train'].append(np.average(train_epoch_loss))

            # =====================valid============================
            self.img_encoder.eval()
            self.img_decoder.eval()
            valid_epoch_loss = []

            for idx, (data_x, data_y), index in enumerate(self.valid_loader, 0):
                data_y = data_y.to(torch.float32).to(self.args['t'].device)
                with torch.no_grad():
                    latent = self.img_encoder(data_y).data
                    output = self.img_decoder(latent)
                    loss = self.args['t'].criterion(output, data_y)
                valid_epoch_loss.append(loss.item())
            self.train_loss['t']['valid'].append(np.average(valid_epoch_loss))

        if autosave:
            torch.save(self.img_encoder.state_dict(),
                       f"../saved/{self.img_encoder}{self.current_title()}_{notion}.pth")
            torch.save(self.img_decoder.state_dict(),
                       f"../saved/{self.img_decoder}{self.current_title()}_{notion}.pth")

    @timer
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

            for idx, (data_x, data_y, index) in enumerate(self.train_loader, 0):
                data_x = data_x.to(torch.float32).to(self.args['s'].device)
                data_y = data_y.to(torch.float32).to(self.args['s'].device)

                student_preds = self.csi_encoder(data_x)
                with torch.no_grad():
                    teacher_preds = self.img_encoder(data_y)
                    image_preds = self.img_decoder(student_preds)

                image_loss = self.img_loss(image_preds, data_y)
                student_loss = self.args['s'].criterion(student_preds, teacher_preds)
                distil_loss = self.div_loss(self.logsoftmax(student_preds / self.temperature),
                                            nn.functional.softmax(teacher_preds / self.temperature, -1))
                loss = self.alpha * student_loss + (1 - self.alpha) * distil_loss

                student_optimizer.zero_grad()
                loss.backward()
                student_optimizer.step()

                train_epoch_loss.append(loss.item())
                straight_epoch_loss.append(student_loss.item())
                distil_epoch_loss.append(distil_loss.item())
                image_epoch_loss.append(image_loss.item())

                if idx % (len(self.train_loader) // 5) == 0:
                    print("\rStudent: epoch={}/{}, {}/{} of train, student loss={}, distill loss={}".format(
                        epoch, self.args['s'].epochs, idx, len(self.train_loader),
                        loss.item(), distil_loss.item()), end='')

            self.train_loss['s']['train'].append(np.average(train_epoch_loss))
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

            for idx, (data_x, data_y, index) in enumerate(self.valid_loader, 0):
                data_x = data_x.to(torch.float32).to(self.args['s'].device)
                data_y = data_y.to(torch.float32).to(self.args['s'].device)
                with torch.no_grad():
                    teacher_preds = self.img_encoder(data_y)
                    student_preds = self.csi_encoder(data_x)
                    image_preds = self.img_decoder(student_preds)
                    image_loss = self.img_loss(image_preds, data_y)
                    student_loss = self.args['s'].criterion(student_preds, teacher_preds)
                    distil_loss = self.div_loss(self.logsoftmax(student_preds / self.temperature),
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

    def test_teacher(self, mode='test'):
        self.test_loss['t'] = self.__gen_teacher_test__()
        self.img_encoder.eval()
        self.img_decoder.eval()

        if mode == 'test':
            loader = self.test_loader
        elif mode == 'train':
            loader = self.train_loader

        for idx, (data_x, data_y, index) in enumerate(loader, 0):
            data_y = data_y.to(torch.float32).to(self.args['t'].device)
            if loader.batch_size != 1:
                data_y = data_y[0][np.newaxis, ...]
            with torch.no_grad():
                latent = self.img_encoder(data_y)
                output = self.img_decoder(latent)
                loss = self.args['s'].criterion(output, data_y)

            self.test_loss['t']['loss'].append(loss.item())
            self.test_loss['t']['predicts'].append(output.cpu().detach().numpy().squeeze())
            self.test_loss['t']['groundtruth'].append(data_y.cpu().detach().numpy().squeeze())

            if idx % (len(loader) // 5) == 0:
                print("\rTeacher: {}/{} of test, loss={}".format(idx, len(self.test_loader), loss.item()), end='')

    def test_student(self, mode='test'):
        self.test_loss['s'] = self.__gen_student_test__()
        self.img_encoder.eval()
        self.img_decoder.eval()
        self.csi_encoder.eval()

        if mode == 'test':
            loader = self.test_loader
        elif mode == 'train':
            loader = self.train_loader

        for idx, (data_x, data_y, index) in enumerate(loader, 0):
            data_x = data_x.to(torch.float32).to(self.args['s'].device)
            data_y = data_y.to(torch.float32).to(self.args['s'].device)
            if loader.batch_size != 1:
                data_x = data_x[0][np.newaxis, ...]
                data_y = data_y[0][np.newaxis, ...]
            with torch.no_grad():
                teacher_preds = self.img_encoder(data_y)
                student_preds = self.csi_encoder(data_x)
                image_preds = self.img_decoder(student_preds)
            student_loss = self.args['s'].criterion(student_preds, teacher_preds)
            image_loss = self.img_loss(image_preds, data_y)
            distil_loss = self.div_loss(self.logsoftmax(student_preds / self.temperature),
                                        nn.functional.softmax(teacher_preds / self.temperature, -1))
            loss = self.alpha * student_loss + (1 - self.alpha) * distil_loss

            self.test_loss['s']['loss'].append(image_loss.item())
            self.test_loss['s']['latent_straight'].append(student_loss.item())
            self.test_loss['s']['latent_distil'].append(loss.item())
            self.test_loss['s']['image'].append(image_loss.item())
            self.test_loss['s']['predicts_latent'].append(student_preds.cpu().detach().numpy().squeeze())
            self.test_loss['s']['predicts_t_latent'].append(teacher_preds.cpu().detach().numpy().squeeze())
            self.test_loss['s']['predicts'].append(image_preds.cpu().detach().numpy().squeeze())
            self.test_loss['s']['groundtruth'].append(data_y.cpu().detach().numpy().squeeze())

            if idx % (len(loader) // 5) == 0:
                print("\rStudent: {}/{}of test, student loss={}, distill loss={}, image loss={}".format(
                    idx, len(self.test_loader), student_loss.item(), distil_loss.item(), image_loss.item()), end='')

    def plot_teacher_loss(self, double_y=False, autosave=False, notion=''):
        self.__plot_settings__()

        loss_items = self.plot_terms['t']['train']
        stage_color = self.colors(self.train_loss['t']['learning_rate'])
        line_color = ['b', 'orange']

        # Training & Validation Loss
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Teacher Training Status @ep{self.train_loss['t']['epochs'][-1]}")
        if len(loss_items.keys()) > 1:
            axes = fig.subplots(1, len(loss_items.keys()))
            axes = axes.flatten()
        else:
            axes = [plt.gca()]

        for i, loss in enumerate(loss_items.keys()):
            for j, learning_rate in enumerate(self.train_loss['t']['learning_rate']):
                axes[i].axvline(self.train_loss['t']['epochs'][j],
                                linestyle='--',
                                color=stage_color[j],
                                label=f'lr={learning_rate}')

            axes[i].plot(list(range(len(self.train_loss['t'][loss_items[loss][1]]))),
                         self.train_loss['t'][loss_items[loss][1]],
                         line_color[1], label=loss_items[loss][1])
            if double_y:
                ax_r = axes[i].twinx()
            else:
                ax_r = axes[i]
            ax_r.plot(list(range(len(self.train_loss['t'][loss_items[loss][0]]))),
                      self.train_loss['t'][loss_items[loss][0]],
                      line_color[0], label=loss_items[loss][0])
            axes[i].set_title(loss)
            axes[i].set_xlabel('#Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].grid()
            axes[i].legend()

        if autosave:
            plt.savefig(f"{self.current_title()}_T_train_{notion}.jpg")
        plt.show()

    def plot_student_loss(self, autosave=False, notion=''):
        self.__plot_settings__()

        loss_items = self.plot_terms['s']['train']
        stage_color = self.colors(-np.log(self.train_loss['s']['learning_rate']))
        line_color = ['b', 'orange']

        # Training & Validation Loss
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Student Training Status @ep{self.train_loss['s']['epochs'][-1]}")
        axes = fig.subplots(nrows=2, ncols=2)
        axes = axes.flatten()

        for i, loss in enumerate(loss_items.keys()):
            for j, learning_rate in enumerate(self.train_loss['t']['learning_rate']):
                axes[i].axvline(self.train_loss['s']['epochs'][j],
                                linestyle='--',
                                color=stage_color[j],
                                label=f'lr={learning_rate}')

            axes[i].plot(self.train_loss['s'][loss_items[loss][1]], line_color[1], label=loss_items[loss][1])
            axes[i].plot(self.train_loss['s'][loss_items[loss][0]], line_color[0], label=loss_items[loss][0])
            axes[i].set_title(loss)
            axes[i].set_xlabel('#Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].grid()

        if autosave:
            plt.savefig(f"{self.current_title()}_S_train_{notion}.jpg")
        plt.show()

    def plot_teacher_test(self, select_ind=None, select_num=8, autosave=False, notion=''):
        self.__plot_settings__()
        predict_items = self.plot_terms['t']['predict']

        # Depth Images
        if select_ind:
            inds = select_ind
        else:
            inds = np.random.choice(list(range(len(self.test_loss['t']['groundtruth']))), select_num, replace=False)
        inds = np.sort(inds)

        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Teacher Test Predicts @ep{self.train_loss['t']['epochs'][-1]}")
        subfigs = fig.subfigures(nrows=2, ncols=1)

        for i, item in enumerate(predict_items.keys()):
            subfigs[i].suptitle(predict_items[item])
            axes = subfigs[i].subplots(nrows=1, ncols=select_num)
            for j in range(len(axes)):
                img = axes[j].imshow(self.test_loss['t'][predict_items[item]][inds[j]], vmin=0, vmax=1)
                axes[j].axis('off')
                axes[j].set_title(f"#{inds[j]}")
            subfigs[i].colorbar(img, ax=axes, shrink=0.8)

        if autosave:
            plt.savefig(f"{self.current_title()}_T_predict_{notion}.jpg")
        plt.show()

        # Test Loss
        loss_items = self.plot_terms['t']['test']
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Teacher Test Loss @ep{self.train_loss['t']['epochs'][-1]}")
        if len(loss_items.keys()) > 1:
            axes = fig.subplots(1, len(loss_items.keys()))
            axes = axes.flatten()
        else:
            axes = [plt.gca()]

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

    def plot_student_test(self, select_num=8, autosave=False, notion=''):
        self.__plot_settings__()
        predict_items = self.plot_terms['s']['predict']

        # Depth Images
        inds = np.random.choice(list(range(len(self.test_loss['s']['groundtruth']))), select_num)
        inds = np.sort(inds)
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Student Test Predicts @ep{self.train_loss['s']['epochs'][-1]}")
        subfigs = fig.subfigures(nrows=2, ncols=1)

        for i, item in enumerate(predict_items.keys()):
            subfigs[i].suptitle(predict_items[item])
            axes = subfigs[i].subplots(nrows=1, ncols=select_num)
            for j in range(len(axes)):
                img = axes[j].imshow(self.test_loss['s'][predict_items[item]][inds[j]], vmin=0, vmax=1)
                axes[j].axis('off')
                axes[j].set_title(f"#{inds[j]}")
            subfigs[i].colorbar(img, ax=axes, shrink=0.8)

        subfigs[1].suptitle('Estimated')

        if autosave:
            plt.savefig(f"{self.current_title()}_S_predict_{notion}.jpg")
        plt.show()

        # Latent Vectors
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Student Test Latents @ep{self.train_loss['s']['epochs'][-1]}")
        axes = fig.subplots(nrows=2, ncols=np.ceil(select_num/2).astype(int))
        axes = axes.flatten()
        for a in range(len(axes)):
            axes[a].bar(range(len(self.test_loss['s']['predicts_t_latent'][inds[a]])),
                        self.test_loss['s']['predicts_t_latent'][inds[a]],
                        width=1, fc='blue', alpha=0.8, label='Teacher')
            axes[a].bar(range(len(self.test_loss['s']['predicts_t_latent'][inds[a]])),
                        self.test_loss['s']['predicts_latent'][inds[a]],
                        width=1, fc='orange', alpha=0.8, label='student')
            axes[a].set_title(f"#{inds[a]}")
            axes[a].grid()

        axes[0].legend()

        if autosave:
            plt.savefig(f"{self.current_title()}_S_latent_{notion}.jpg")
        plt.show()

        # Test Loss
        loss_items = self.plot_terms['s']['test']
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Student Test Loss @ep{self.train_loss['s']['epochs'][-1]}")
        axes = fig.subplots(nrows=2, ncols=2)
        axes = axes.flatten()

        for i, loss in enumerate(loss_items.keys()):
            axes[i].scatter(list(range(len(self.test_loss['s']['groundtruth']))),
                            self.test_loss['s'][loss_items[loss]], alpha=0.6)
            axes[i].set_title(loss)
            axes[i].set_xlabel('#Sample')
            axes[i].set_ylabel('Loss')
            axes[i].grid()
            for j in inds:
                axes[i].scatter(j, self.test_loss['s'][loss_items[loss]][j],
                                c='magenta', marker=(5, 1), linewidths=4)
        if autosave:
            plt.savefig(f"{self.current_title()}_S_test_{notion}.jpg")
        plt.show()

    def traverse_latent(self, img_ind, dataset, mode='t', img='x', dim1=0, dim2=1, granularity=11, autosave=False, notion=''):
        self.__plot_settings__()

        self.img_encoder.eval()
        self.img_decoder.eval()
        self.csi_encoder.eval()

        if img_ind >= len(dataset):
            img_ind = np.random.randint(len(dataset))

        try:
            data_x, data_y = dataset[img_ind]
            if img == 'x':
                image = data_x[np.newaxis, ...]
            elif img == 'y':
                image = data_y[np.newaxis, ...]
                csi = data_x[np.newaxis, ...]

        except ValueError:
            image = dataset[img_ind][np.newaxis, ...]

        if mode == 't':
            z = self.img_encoder(torch.from_numpy(image).to(torch.float32).to(self.args['t'].device))
        if mode == 's':
            z = self.img_encoder(torch.from_numpy(csi).to(torch.float32).to(self.args['s'].device))

        z = z.cpu().detach().numpy().squeeze()

        grid_x = np.linspace(np.min(z), np.max(z), granularity)
        grid_y = np.linspace(np.min(z), np.max(z), granularity)
        anchor1 = np.searchsorted(grid_x, z[dim1])
        anchor2 = np.searchsorted(grid_y, z[dim2])
        anchor1 = anchor1 * 128 if anchor1 < granularity else (anchor1 - 1) * 128
        anchor2 = anchor2 * 128 if anchor2 < granularity else (anchor2 - 1) * 128

        figure = np.zeros((granularity * 128, granularity * 128))

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z[dim1], z[dim2] = xi, yi
                output = self.img_decoder(torch.from_numpy(z).to(self.args['t'].device))
                figure[i * 128: (i + 1) * 128,
                       j * 128: (j + 1) * 128] = output.cpu().detach().numpy().squeeze()

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

    def save_all_params(self, notion=''):
        save_path = '/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(self.img_encoder.state_dict(),
                   f"../saved/{self.img_encoder}{self.current_title()}_{notion}.pth")
        torch.save(self.img_decoder.state_dict(),
                   f"../saved/{self.img_decoder}{self.current_title()}_{notion}.pth")
        torch.save(self.csi_encoder.state_dict(),
                   f"../saved/{self.csi_encoder}{self.current_title()}_{notion}.pth")
