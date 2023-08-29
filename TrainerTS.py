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
    def __init__(self, x_path, y_path, img_size=(128, 128), transform=None, img='y', int_image=False, number=0):
        self.seeds = None
        self.img_size = img_size
        self.transform = transform
        self.img = img
        self.int_img = int_image
        self.data = self.__load_data__(x_path, y_path, number=number)
        print('loaded')

    def __convert2int(self, sample):
        if self.int_img:
            return np.uint8(np.array(sample * 255))
        else:
            return sample

    def __transform__(self, sample):
        if self.transform:
            return self.transform(Image.fromarray(sample).squeeze(), mode='L')
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
        self.valid_loss = {'t': self.__gen_teacher_train__(),
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

        self.temp_loss = {}

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
                        'LOSS': [],
                        }
        return t_train_loss

    @staticmethod
    def __gen_student_train__():
        s_train_loss = {'learning_rate': [],
                        'epochs': [0],
                        'LOSS': [],
                        'STRA': [],
                        'DIST': [],
                        'IMG': [],
                        }
        return s_train_loss

    @staticmethod
    def __gen_teacher_test__():
        t_test_loss = {'loss': [],
                       'PRED': [],
                       'GT': [],
                       'IND': []
                       }
        return t_test_loss

    @staticmethod
    def __gen_student_test__():
        s_test_loss = {'LOSS': [],
                       'STRA': [],
                       'DIST': [],
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
    def __teacher_plot_terms__():
        terms = {'loss': {'LOSS': 'Loss'},
                 'predict': ('GT', 'PRED', 'IND'),
                 'test': {'GT': 'Ground Truth',
                          'PRED': 'Estimated'
                          }
                 }
        return terms

    @staticmethod
    def __student_plot_terms__():
        terms = {'loss': {'LOSS': 'Loss',
                          'STRA': 'Straight',
                          'DIST': 'Distilation',
                          'IMG': 'Image'},
                 'predict': ('GT', 'S_PRED', 'T_PRED', 'T_LATENT', 'S_LATENT', 'IND'),
                 'test': {'GT': 'Ground Truth',
                          'T_PRED': 'Teacher Estimate',
                          'S_PRED': 'Student Estimate'}
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
        objs = (self.train_loss, self.valid_loss)
        for obj in objs:

            # First round
            if not obj[mode]['learning_rate']:
                obj[mode]['learning_rate'].append(self.args[mode].learning_rate)
                obj[mode]['epochs'].append(self.args[mode].epochs)

            else:
                # Not changing learning rate
                if self.args[mode].learning_rate == obj[mode]['learning_rate'][-1]:
                    obj[mode]['epochs'][-1] += self.args[mode].epochs

                # Changing learning rate
                if self.args[mode].learning_rate != obj[mode]['learning_rate'][-1]:
                    last_end = self.train_loss[mode]['epochs'][-1]
                    obj[mode]['learning_rate'].append(self.args[mode].learning_rate)
                    obj[mode]['epochs'].append(last_end + self.args[mode].epochs)

    def calculate_loss(self, mode, x, y, i=None):
        if mode == 't':
            latent = self.img_encoder(y)
            output = self.img_decoder(latent)
            loss = self.args[mode].criterion(output, y)
            self.temp_loss = {'LOSS': loss}
            return {'GT': y,
                    'PRED': output,
                    'IND': i}

        elif mode == 's':
            s_latent = self.csi_encoder(x)
            with torch.no_grad():
                t_latent = self.img_encoder(y)
                s_output = self.img_decoder(s_latent)
                t_output = self.img_decoder(t_latent)
                image_loss = self.img_loss(s_output, y)

            straight_loss = self.args[mode].criterion(s_latent, t_latent)
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
        self.logger(mode='t')
        teacher_optimizer = self.args['t'].optimizer([{'params': self.img_encoder.parameters()},
                                                      {'params': self.img_decoder.parameters()}],
                                                     lr=self.args['t'].learning_rate)
        LOSS_TERMS = self.plot_terms['t']['loss'].keys()

        for epoch in range(self.args['t'].epochs):

            # =====================train============================
            self.img_encoder.train()
            self.img_decoder.train()
            EPOCH_LOSS = {key: [] for key in LOSS_TERMS}
            for idx, (data_x, data_y, index) in enumerate(self.train_loader, 0):
                data_y = data_y.to(torch.float32).to(self.args['t'].device)
                teacher_optimizer.zero_grad()

                PREDS = self.calculate_loss('t', None, data_y)
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
            torch.save(self.img_encoder.state_dict(),
                       f"../saved/{notion}_{self.img_encoder}_{self.current_title()}.pth")
            torch.save(self.img_decoder.state_dict(),
                       f"../saved/{notion}_{self.img_decoder}_{self.current_title()}.pth")

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
            LOSS_TERMS = self.plot_terms['s']['loss'].keys()
            EPOCH_LOSS = {key: [] for key in LOSS_TERMS}

            for idx, (data_x, data_y, index) in enumerate(self.train_loader, 0):
                data_x = data_x.to(torch.float32).to(self.args['s'].device)
                data_y = data_y.to(torch.float32).to(self.args['s'].device)

                PREDS = self.calculate_loss('s', data_x, data_y)
                student_optimizer.zero_grad()
                self.temp_loss['LOSS'].backward()
                student_optimizer.step()

                for key in LOSS_TERMS:
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

                if idx % (len(self.train_loader) // 5) == 0:
                    print(f"\rStudent: epoch={epoch}/{self.args['t'].epochs}, batch={idx}/{len(self.train_loader)},"
                          f"loss={self.temp_loss['LOSS'].item()}", end='')

            for key in LOSS_TERMS:
                self.train_loss['s'][key].append(np.average(EPOCH_LOSS[key]))

            # =====================valid============================
            self.csi_encoder.eval()
            self.img_encoder.eval()
            self.img_decoder.eval()
            EPOCH_LOSS = {key: [] for key in LOSS_TERMS}

            for idx, (data_x, data_y, index) in enumerate(self.valid_loader, 0):
                data_x = data_x.to(torch.float32).to(self.args['s'].device)
                data_y = data_y.to(torch.float32).to(self.args['s'].device)
                with torch.no_grad():
                    PREDS = self.calculate_loss('s', data_x, data_y)

                for key in LOSS_TERMS:
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

            for key in LOSS_TERMS:
                self.valid_loss['s'][key].append(np.average(EPOCH_LOSS[key]))

        if autosave:
            torch.save(self.csi_encoder.state_dict(),
                       f"../saved/{notion}_{self.csi_encoder}_{self.current_title()}.pth")

    def test_teacher(self, mode='test'):
        self.test_loss['t'] = self.__gen_teacher_test__()
        self.img_encoder.eval()
        self.img_decoder.eval()
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
                    PREDS = self.calculate_loss('t', None, gt, ind)

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

    def test_student(self, mode='test'):
        self.test_loss['s'] = self.__gen_student_test__()
        self.img_encoder.eval()
        self.img_decoder.eval()
        self.csi_encoder.eval()
        LOSS_TERMS = self.plot_terms['s']['loss'].keys()
        PLOT_TERMS = self.plot_terms['s']['predict']
        EPOCH_LOSS = {key: [] for key in LOSS_TERMS}

        if mode == 'test':
            loader = self.test_loader
        elif mode == 'train':
            loader = self.train_loader

        for idx, (data_x, data_y, index) in enumerate(loader, 0):
            data_x = data_x.to(torch.float32).to(self.args['s'].device)
            data_y = data_y.to(torch.float32).to(self.args['s'].device)
            with torch.no_grad():
                for sample in range(loader.batch_size):
                    ind = index[sample][np.newaxis, ...]
                    csi = data_x[sample][np.newaxis, ...]
                    gt = data_y[sample][np.newaxis, ...]
                    PREDS = self.calculate_loss('s', csi, gt, ind)

                    for key in LOSS_TERMS:
                        EPOCH_LOSS[key].append(self.temp_loss[key].item())

                    for key in PLOT_TERMS:
                        self.test_loss['s'][key].append(PREDS[key].cpu().detach().numpy().squeeze())

            for key in LOSS_TERMS:
                self.test_loss['s'][key] = EPOCH_LOSS[key]

            if idx % (len(loader) // 5) == 0:
                print(f"\rStudent: test={idx}/{len(loader)}, loss={self.temp_loss['LOSS'].item()}", end='')

        for key in LOSS_TERMS:
            EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
        print(f"\nTest finished. Average loss={EPOCH_LOSS}")

    def plot_train_loss(self, mode, double_y=False, autosave=False, notion=''):
        self.__plot_settings__()

        PLOT_ITEMS = self.plot_terms[mode]['loss']
        stage_color = self.colors(self.train_loss[mode]['learning_rate'])
        line_color = ['b', 'orange']
        epoch = self.train_loss[mode]['epochs'][-1]
        save_path = f'../saved/{notion}/'

        title = {'t': f"Teacher Training Status @ep{epoch}",
                 's': f"Student Training Status @ep{epoch}"}
        filename = {'t': f"{save_path}{notion}_T_train_{self.current_title()}.jpg",
                    's': f"{save_path}{notion}_S_train_{self.current_title()}.jpg"}

        # Training & Validation Loss
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(title[mode])
        if len(PLOT_ITEMS.keys()) == 1:
            axes = [plt.gca()]
        elif len(PLOT_ITEMS.keys()) > 3:
            axes = fig.subplots(2, np.ceil(len(PLOT_ITEMS.keys())/2).astype(int))
            axes = axes.flatten()
        else:
            axes = fig.subplots(1, len(PLOT_ITEMS.keys()))
            axes = axes.flatten()

        for i, loss in enumerate(PLOT_ITEMS.keys()):
            for j, learning_rate in enumerate(self.train_loss[mode]['learning_rate']):
                axes[i].axvline(self.train_loss[mode]['epochs'][j],
                                linestyle='--',
                                color=stage_color[j],
                                label=f'lr={learning_rate}')

            axes[i].plot(list(range(len(self.valid_loss[mode][loss]))),
                         self.valid_loss[mode][loss],
                         line_color[1], label='Valid')
            if double_y:
                ax_r = axes[i].twinx()
            else:
                ax_r = axes[i]
            ax_r.plot(list(range(len(self.train_loss[mode][loss]))),
                      self.train_loss[mode][loss],
                      line_color[0], label='Train')
            axes[i].set_title(PLOT_ITEMS[loss])
            axes[i].set_xlabel('#Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].grid()
            axes[i].legend()

        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(filename[mode])
        plt.show()

    def plot_test(self, mode, select_ind=None, select_num=8, autosave=False, notion=''):
        self.__plot_settings__()
        PLOT_ITEMS = self.plot_terms[mode]['test']
        LOSS_ITEMS = self.plot_terms[mode]['loss']
        epoch = self.train_loss[mode]['epochs'][-1]
        save_path = f'../saved/{notion}/'

        title = {'t': {'PRED': f"Teacher Test Predicts @ep{epoch}",
                       'LOSS': f"Teacher Test Loss @ep{epoch}"},
                 's': {'PRED': f"Student Test Predicts @ep{epoch}",
                       'LOSS': f"Student Test Loss @ep{epoch}",
                       'LATENT': f"Student Test Latents @ep{epoch}"}}
        filename = {'t': {'PRED': f"{save_path}{notion}_T_predict_{self.current_title()}.jpg",
                          'LOSS': f"{save_path}{notion}_T_test_{self.current_title()}.jpg"},
                    's': {'PRED': f"{save_path}{notion}_S_predict_{self.current_title()}.jpg",
                          'LOSS': f"{save_path}{notion}_S_test_{self.current_title()}.jpg",
                          'LATENT': f"{save_path}{notion}_S_latent_{self.current_title()}.jpg"}}

        if select_ind:
            inds = select_ind
        else:
            inds = np.random.choice(list(range(len(self.test_loss[mode]['IND']))), select_num, replace=False)
        inds = np.sort(inds)
        samples = np.array(self.test_loss[mode]['IND'])[inds]

        # Depth Images
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(title[mode]['PRED'])
        subfigs = fig.subfigures(nrows=len(PLOT_ITEMS.keys()), ncols=1)

        for i, item in enumerate(PLOT_ITEMS.keys()):
            subfigs[i].suptitle(PLOT_ITEMS[item])
            axes = subfigs[i].subplots(nrows=1, ncols=select_num)
            for j in range(len(axes)):
                img = axes[j].imshow(self.test_loss[mode][item][inds[j]], vmin=0, vmax=1)
                axes[j].axis('off')
                axes[j].set_title(f"#{samples[j]}")
            subfigs[i].colorbar(img, ax=axes, shrink=0.8)

        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(filename[mode]['PRED'])
        plt.show()

        # Test Loss
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(title[mode]['LOSS'])
        if len(LOSS_ITEMS.keys()) == 1:
            axes = [plt.gca()]
        elif len(LOSS_ITEMS.keys()) > 3:
            axes = fig.subplots(2, np.ceil(len(LOSS_ITEMS.keys())/2).astype(int))
            axes = axes.flatten()
        else:
            axes = fig.subplots(1, len(LOSS_ITEMS.keys()))
            axes = axes.flatten()

        for i, item in enumerate(LOSS_ITEMS.keys()):
            axes[i].scatter(list(range(len(self.test_loss[mode][item]))),
                            self.test_loss[mode][item], alpha=0.6)
            axes[i].set_title(LOSS_ITEMS[item])
            axes[i].set_xlabel('#Sample')
            axes[i].set_ylabel('Loss')
            axes[i].grid()
            for j in range(select_num):
                axes[i].scatter(inds[j], self.test_loss[mode][item][inds[j]],
                                c='magenta', marker=(5, 1), linewidths=4)

        if autosave:
            plt.savefig(filename[mode]['LOSS'])
        plt.show()

        # Latent Vectors
        if mode == 's':
            fig = plt.figure(constrained_layout=True)
            fig.suptitle(title[mode]['LATENT'])
            axes = fig.subplots(nrows=2, ncols=np.ceil(select_num / 2).astype(int))
            axes = axes.flatten()
            for j in range(select_num):
                axes[j].bar(range(len(self.test_loss[mode]['T_LATENT'][inds[0]])),
                            self.test_loss[mode]['T_LATENT'][inds[j]],
                            width=1, fc='blue', alpha=0.8, label='Teacher')
                axes[j].bar(range(len(self.test_loss[mode]['S_LATENT'][inds[0]])),
                            self.test_loss[mode]['S_LATENT'][inds[j]],
                            width=1, fc='orange', alpha=0.8, label='Student')
                axes[j].set_title(f"#{samples[j]}")
                axes[j].grid()

            axes[0].legend()

            if autosave:
                plt.savefig(filename[mode]['LATENT'])
            plt.show()

    def traverse_latent(self, img_ind, dataset, mode='t', img='x', dim1=0, dim2=1, granularity=11, autosave=False, notion=''):
        self.__plot_settings__()

        self.img_encoder.eval()
        self.img_decoder.eval()
        self.csi_encoder.eval()

        if img_ind >= len(dataset):
            img_ind = np.random.randint(len(dataset))

        try:
            data_x, data_y, index = dataset[img_ind]
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
            save_path = f'../saved/{notion}/'

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.savefig(f"{save_path}{notion}_T_traverse_{dim1}{dim2}_{self.current_title()}.jpg")
        plt.show()

    def save_all_params(self, notion=''):
        save_path = f'../saved/{notion}/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(self.img_encoder.state_dict(),
                   f"{save_path}{notion}_{self.img_encoder}{self.current_title()}.pth")
        torch.save(self.img_decoder.state_dict(),
                   f"{save_path}{notion}_{self.img_decoder}{self.current_title()}.pth")
        torch.save(self.csi_encoder.state_dict(),
                   f"{save_path}{notion}_{self.csi_encoder}{self.current_title()}.pth")

    def scheduler(self, train_t=True, train_s=True, t_turns=10, s_turns=10, lr_decay=False, decay_rate=0.4, test_mode='train', autosave=False, notion=''):

        if train_t:
            for i in range(t_turns):
                self.train_teacher()
                self.test_teacher(mode=test_mode)
                self.plot_test(mode='t', autosave=autosave, notion=notion)
                self.plot_train_loss(mode='t', autosave=autosave, notion=notion)
                if lr_decay:
                    self.args['t'].learning_rate *= decay_rate

        if train_s:
            for i in range(s_turns):
                self.train_student()
                self.test_student(mode=test_mode)
                self.plot_test(mode='s', autosave=autosave, notion=notion)
                self.plot_train_loss(mode='s', autosave=autosave, notion=notion)
                if lr_decay:
                    self.args['s'].learning_rate *= decay_rate

        print("\nSchedule Completed!")
