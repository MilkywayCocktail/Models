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
    def __init__(self, x_path, y_path, transform=None, number=0):
        self.seeds = None
        self.transform = transform
        self.data = self.load_data(x_path, y_path, number=number)
        print('loaded')

    def __getitem__(self, index):
        if self.transform:
            image = self.transform(Image.fromarray((np.array(self.data['y'][index])).squeeze(), mode='L'))
        else:
            image = self.data['y'][index]

        return self.data['x'][index], image

    def __len__(self):
        return self.data['x'].shape[0]

    def load_data(self, x_path, y_path, number):
        x = np.load(x_path)
        y = np.load(y_path)

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


def split_loader(dataset, train_size, valid_size, test_size, batch_size):
    train_dataset, valid_dataset, test_dataset = Data.random_split(dataset, [train_size, valid_size, test_size])
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = Data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    return train_loader, valid_loader, test_loader


class MyArgs:
    def __init__(self, cuda=1, epochs=30, learning_rate=0.001, criterion=nn.CrossEntropyLoss()):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda:" + str(cuda) if torch.cuda.is_available() else "cpu")
        self.criterion = criterion


class TrainerTeacherStudent:

    def __init__(self, img_encoder, img_decoder, csi_encoder,
                 teacher_args, student_args,
                 train_loader, valid_loader, test_loader,
                 optimizer=torch.optim.Adam,
                 div_loss=nn.KLDivLoss(reduction='batchmean'),
                 img_loss=nn.SmoothL1Loss(),
                 temperature=20,
                 alpha=0.3):
        self.img_encoder = img_encoder
        self.img_decoder = img_decoder
        self.csi_encoder = csi_encoder

        self.args = {'t': teacher_args,
                     's': student_args}

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.teacher_optimizer = optimizer([{'params': self.img_encoder.parameters()},
                                           {'params': self.img_decoder.parameters()}],
                                           lr=self.args['t'].learning_rate)
        self.student_optimizer = optimizer(self.csi_encoder.parameters(), lr=self.args['s'].learning_rate)

        self.train_loss = {'t': self.__gen_teacher_train__(),
                           's': self.__gen_student_train__()}
        self.test_loss = {'t': self.__gen_teacher_test__(),
                          's': self.__gen_student_test__()}
        self.plot_terms = {
            't_train': {'Loss': ['train', 'valid']
                        },
            't_predict': {'Ground Truth': 'groundtruth',
                          'Estimated': 'predicts'
                          },
            't_test': {'Loss': 'loss'}
        }

        self.div_loss = div_loss
        self.temperature = temperature
        self.alpha = alpha
        self.img_loss = img_loss

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
                        'epochs': [],
                        'train': [],
                        'valid': [],
                        }
        return t_train_loss

    @staticmethod
    def __gen_student_train__():
        s_train_loss = {'learning_rate': [],
                        'epochs': [],
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

    def current_title(self):
        return f"Te{self.train_loss['t']['epochs'][-1]}_Se{self.train_loss['s']['epochs'][-1]}"

    @staticmethod
    def colors(arrays):
        arr = np.array(arrays)
        norm = plt.Normalize(arr.min(), arr.max())
        map_vir = cm.get_cmap(name='viridis')
        c = map_vir(norm(arr))
        return c

    def logger(self, mode='t'):
        # First round
        if not self.train_loss[mode]['learning_rate']:
            self.train_loss[mode]['learning_rate'].append(self.args[mode].learning_rate)
            self.train_loss[mode]['epochs'].append(0)
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

        for epoch in range(self.args['t'].epochs):

            # =====================train============================
            self.img_encoder.train()
            self.img_decoder.train()
            train_epoch_loss = []
            for idx, (data_x, data_y) in enumerate(self.train_loader, 0):
                data_y = data_y.to(torch.float32).to(self.args['t'].device)
                self.teacher_optimizer.zero_grad()
                latent = self.img_encoder(data_y).data
                output = self.img_decoder(latent)

                loss = self.args['t'].criterion(output, data_y)
                loss.backward()
                self.teacher_optimizer.step()
                train_epoch_loss.append(loss.item())

                if idx % (len(self.train_loader) // 5) == 0:
                    print("\rTeacher: epoch={}/{}, {}/{} of train, loss={}".format(
                        epoch, self.args['t'].epochs, idx, len(self.train_loader), loss.item()), end='')
            self.train_loss['t']['train'].append(np.average(train_epoch_loss))

            # =====================valid============================
            self.img_encoder.eval()
            self.img_decoder.eval()
            valid_epoch_loss = []

            for idx, (data_x, data_y) in enumerate(self.valid_loader, 0):
                data_y = data_y.to(torch.float32).to(self.args['t'].device)
                with torch.no_grad():
                    latent = self.img_encoder(data_y).data
                    output = self.img_decoder(latent)
                    loss = self.args['t'].criterion(output, data_y)
                valid_epoch_loss.append(loss.item())
            self.train_loss['t']['valid'].append(np.average(valid_epoch_loss))

        if autosave:
            torch.save(self.img_encoder.state_dict(),
                       f"../Models/{self.img_encoder}{self.current_title()}_{notion}.pth")
            torch.save(self.img_decoder.state_dict(),
                       f"../Models/{self.img_decoder}{self.current_title()}_{notion}.pth")

    @timer
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

                student_preds = self.csi_encoder(data_x)
                with torch.no_grad():
                    teacher_preds = self.img_encoder(data_y)
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
                    print("\rStudent: epoch={}/{}, {}/{} of train, student loss={}, distill loss={}".format(
                        epoch, self.args['s'].epochs, idx, len(self.train_loader),
                        loss.item(), distil_loss.item()), end='')

            self.train_loss['s']['train_epochs'].append(np.average(train_epoch_loss))
            self.train_loss['s']['train_straight_epochs'].append(np.average(straight_epoch_loss))
            self.train_loss['s']['train_distil_epochs'].append(np.average(distil_epoch_loss))
            self.train_loss['s']['train_image_epochs'].append(np.average(image_epoch_loss))

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

        for idx, (data_x, data_y) in enumerate(loader, 0):
            data_x = data_x.to(torch.float32).to(self.args['s'].device)
            data_y = data_y.to(torch.float32).to(self.args['s'].device)
            with torch.no_grad():
                teacher_latent_preds = self.img_encoder(data_y)
                student_latent_preds = self.csi_encoder(data_x)
                student_image_preds = self.img_decoder(student_latent_preds)
            student_loss = self.args['s'].criterion(student_latent_preds, teacher_latent_preds)
            image_loss = self.img_loss(student_image_preds, data_y)

            distil_loss = self.div_loss(nn.functional.softmax(student_latent_preds / self.temperature, -1),
                                        nn.functional.softmax(teacher_latent_preds / self.temperature, -1))

            loss = self.alpha * student_loss + (1 - self.alpha) * distil_loss

            self.test_loss['s']['loss'].append(image_loss.item())
            self.test_loss['s']['latent_straight'].append(student_loss.item())
            self.test_loss['s']['latent_distil'].append(loss.item())
            self.test_loss['s']['image'].append(image_loss.item())
            self.test_loss['s']['predicts_latent'].append(student_latent_preds.cpu().detach().numpy().squeeze())
            self.test_loss['s']['predicts_t_latent'].append(teacher_latent_preds.cpu().detach().numpy().squeeze())
            self.test_loss['s']['predicts'].append(student_image_preds.cpu().detach().numpy().squeeze())
            self.test_loss['s']['groundtruth'].append(data_y.cpu().detach().numpy().squeeze())

            if idx % (len(loader) // 5) == 0:
                print("\rStudent: {}/{}of test, student loss={}, distill loss={}, image loss={}".format(
                    idx, len(self.test_loader), student_loss.item(), distil_loss.item(), image_loss.item()), end='')

    def plot_teacher_loss(self, autosave=False, notion=''):
        self.__plot_settings__()

        loss_items = self.plot_terms['t_train']
        stage_color = self.colors(self.train_loss['t']['learning_rate'])
        line_color = ['b', 'orange']

        # Training & Validation Loss
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Teacher Training Status @ep{self.train_loss['t']['epochs'][-1]}")
        axes = fig.subplots(1, len(loss_items.keys()))
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

    def plot_student_loss(self, autosave=False, notion=''):
        self.__plot_settings__()

        loss_items = {'Student Loss': ['train', 'valid'],
                      'Straight Loss': ['train_straight', 'valid_straight'],
                      'Distillation Loss': ['train_distil', 'valid_distil'],
                      'Image Loss': ['train_image', 'valid_image']
                      }
        stage_color = self.colors(self.train_loss['s']['learning_rate'])
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
        predict_items = self.plot_terms['t_predict']

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
        loss_items = self.plot_terms['t_test']
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(f"Teacher Test Loss @ep{self.train_loss['t']['epochs'][-1]}")
        axes = fig.subplots(nrows=1, ncols=len(loss_items.keys()))

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
        predict_items = {'Ground Truth': 'groundtruth',
                         'Estimated': 'predicts_image'
                         }

        # Depth Images
        inds = np.random.choice(list(range(len(self.test_loss['s']['groundtruth']))), 8)
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
        axes = fig.subplots(nrows=2, ncols=np.ceil(select_num/2))
        axes = axes.flatten()
        for a in range(len(axes)):
            axes[a].bar(range(256), self.test_loss['s']['predicts_t_latent'][inds[a]], width=1, fc='blue',
                        alpha=0.8, label='Teacher')
            axes[a].bar(range(256), self.test_loss['s']['predicts_latent'][inds[a]], width=1, fc='orange',
                        alpha=0.8, label='student')
            axes[a].set_title(f"#{inds[a]}")
            axes[a].grid()

        axes[0].legend()

        if autosave:
            plt.savefig(f"{self.current_title()}_S_latent_{notion}.jpg")
        plt.show()

        # Test Loss
        loss_items = {'Student Loss': 'loss',
                      'Straight Loss': 'latent_straight',
                      'Distillation Loss': 'latent_distil',
                      'Image Loss': 'image'
                      }
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

    def save_all_params(self, notion=''):
        save_path = '/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(self.img_encoder.state_dict(),
                   f"../Models/{self.img_encoder}{self.current_title()}_{notion}.pth")
        torch.save(self.img_decoder.state_dict(),
                   f"../Models/{self.img_decoder}{self.current_title()}_{notion}.pth")
        torch.save(self.csi_encoder.state_dict(),
                   f"../Models/{self.csi_encoder}{self.current_title()}_{notion}.pth")
