import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm
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


class MyArgs:
    """
    ARGUMENTS
    """
    def __init__(self, cuda=1, epochs=30, learning_rate=0.001,
                 criterion=nn.CrossEntropyLoss(),
                 optimizer=torch.optim.Adam):
        """
        Wraps hyperparameters.
        :param cuda: cuda index
        :param epochs: expected training epochs
        :param learning_rate: learning rate
        :param criterion: loss function
        :param optimizer: default is Adam
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda:" + str(cuda) if torch.cuda.is_available() else "cpu")
        self.criterion = criterion
        self.optimizer = optimizer


class TrainerTS:
    """
    TRAINER FOR TEACHER-STUDENT MODELS
    """
    def __init__(self, img_encoder, img_decoder, csi_encoder,
                 teacher_args, student_args,
                 train_loader, valid_loader, test_loader,
                 div_loss=nn.KLDivLoss(reduction='batchmean'),
                 img_loss=nn.MSELoss(reduction='sum'),
                 temperature=1,
                 alpha=0.3,
                 latent_dim=16):
        """
        Used in Teacher-Student model training
        :param img_encoder: image encoder model
        :param img_decoder: image decoder model
        :param csi_encoder: csi encoder model
        :param teacher_args: teacher's arguments. MyArgs object
        :param student_args: student's arguments. MyArgs object
        :param train_loader: train dataloader
        :param valid_loader: validation dataloader
        :param test_loader: test dataloader
        :param div_loss: divergence loss. Default is KLDivLoss
        :param img_loss: image loss. Not back propagated.
        Only used as a metric. Default is MSE.
        :param temperature: temperature in knowledge distillation. Default is 20
        :param alpha: weight of divergence loss. Default is 0.3
        :param latent_dim: length of latent vector. Default is 8
        """

        self.models = {'imgen': img_encoder,
                       'imgde': img_decoder,
                       'csien': csi_encoder}

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
        """
        Prepares plot configurations.
        :return: plt args
        """
        plt.rcParams['figure.figsize'] = (20, 10)
        plt.rcParams["figure.titlesize"] = 35
        plt.rcParams['lines.markersize'] = 10
        plt.rcParams['axes.titlesize'] = 30
        plt.rcParams['axes.labelsize'] = 30
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20

    @staticmethod
    def __gen_teacher_train__():
        """
        Generates teacher's training loss.
        :return: structured loss
        """
        t_train_loss = {'learning_rate': [],
                        'epochs': [0],
                        'LOSS': [],
                        }
        return t_train_loss

    @staticmethod
    def __gen_student_train__():
        """
        Generates student's training loss.
        :return: structured loss
        """
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
        """
        Generates teacher's test loss.
        :return: structured loss
        """
        t_test_loss = {'loss': [],
                       'PRED': [],
                       'GT': [],
                       'IND': []
                       }
        return t_test_loss

    @staticmethod
    def __gen_student_test__():
        """
        Generates student's test loss.
        :return: structured loss
        """
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
        """
        Defines plot items for plot_test(mode='t')
        :return: keywords
        """
        terms = {'loss': {'LOSS': 'Loss'},
                 'predict': ('GT', 'PRED', 'IND'),
                 'test': {'GT': 'Ground Truth',
                          'PRED': 'Estimated'
                          }
                 }
        return terms

    @staticmethod
    def __student_plot_terms__():
        """
        Defines plot items for plot_test(mode='s')
        :return: keywords
        """
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

    def __train_models_t__(self):
        """
        Changes teacher model states for training.
        :return: None
        """
        self.models['imgen'].train()
        self.models['imgde'].train()
        return [{'params': self.models['imgen'].parameters()},
                {'params': self.models['imgde'].parameters()}]

    def __test_models_t__(self):
        """
        Changes teacher model states for testing.
        :return: None
        """
        self.models['imgen'].eval()
        self.models['imgde'].eval()

    def __train_models_s__(self):
        """
        Changes student model states for training.
        :return: None
        """
        self.models['imgen'].eval()
        self.models['imgde'].eval()
        self.models['csien'].train()
        return [{'params': self.models['csien'].parameters()}]

    def __test_models_s__(self):
        """
        Changes student model states for testing.
        :return: None
        """
        self.models['imgen'].eval()
        self.models['imgde'].eval()
        self.models['csien'].eval()

    def current_title(self):
        """
        Shows current title
        :return: a string including current training epochs
        """
        return f"Te{self.train_loss['t']['epochs'][-1]}_Se{self.train_loss['s']['epochs'][-1]}"

    @staticmethod
    def colors(arrays):
        """
        Color solution for plotting loss curves
        :param arrays: array of learning rates
        :return: a variation of colors
        """
        arr = -np.log(arrays)
        norm = plt.Normalize(arr.min(), arr.max())
        map_vir = cm.get_cmap(name='viridis')
        c = map_vir(norm(arr))
        return c

    def logger(self, mode='t'):
        """
        Logs learning rate and number of epochs before training.
        :param mode: 't' or 's'
        :return: logger decorator
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

    def calculate_loss_t(self, x, y, i=None):
        """
        Calculates loss function for back propagation.
        :param x: x data (CSI)
        :param y: y data (image)
        :param i: index of data
        :return: loss object
        """
        latent = self.models['imgen'](y)
        output = self.models['imgde'](latent)
        loss = self.args['t'].criterion(output, y)
        self.temp_loss = {'LOSS': loss}
        return {'GT': y,
                'PRED': output,
                'IND': i}

    def calculate_loss_s(self, x, y, i=None):
        """
        Calculates loss function for back propagation.
        :param x: x data (CSI)
        :param y: y data (image)
        :param i: index of data
        :return: loss object
        """
        s_latent = self.models['csien'](x)
        with torch.no_grad():
            t_latent = self.models['imgen'](y)
            s_output = self.models['imgde'](s_latent)
            t_output = self.models['imgde'](t_latent)
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
        LOSS_TERMS = self.plot_terms['t']['loss'].keys()
        t_optimizer = self.args['t'].optimizer(self.__train_models_t__(), lr=self.args['t'].learning_rate)

        for epoch in range(self.args['t'].epochs):

            # =====================train============================
            params = self.__train_models_t__()
            EPOCH_LOSS = {key: [] for key in LOSS_TERMS}
            for idx, (data_x, data_y, index) in enumerate(self.train_loader, 0):
                data_y = data_y.to(torch.float32).to(self.args['t'].device)
                t_optimizer.zero_grad()

                PREDS = self.calculate_loss_t(None, data_y)
                self.temp_loss['LOSS'].backward()
                t_optimizer.step()
                for key in LOSS_TERMS:
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

                if idx % (len(self.train_loader) // 5) == 0:
                    print(f"\rTeacher: epoch={epoch}/{self.args['t'].epochs}, batch={idx}/{len(self.train_loader)}, "
                          f"loss={self.temp_loss['LOSS'].item()}", end='')
            for key in LOSS_TERMS:
                self.train_loss['t'][key].append(np.average(EPOCH_LOSS[key]))

            # =====================valid============================
            self.__test_models_t__()
            EPOCH_LOSS = {key: [] for key in LOSS_TERMS}

            for idx, (data_x, data_y, index) in enumerate(self.valid_loader, 0):
                data_y = data_y.to(torch.float32).to(self.args['t'].device)
                with torch.no_grad():
                    PREDS = self.calculate_loss_t(None, data_y)
                for key in LOSS_TERMS:
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())
            for key in LOSS_TERMS:
                self.valid_loss['t'][key].append(np.average(EPOCH_LOSS[key]))

        if autosave:
            save_path = f'../saved/{notion}/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.models['imgen'].state_dict(),
                       f"{save_path}{notion}_{self.models['imgen']}_{self.current_title()}.pth")
            torch.save(self.models['imgde'].state_dict(),
                       f"{save_path}{notion}_{self.models['imgde']}_{self.current_title()}.pth")

    @timer
    def train_student(self, autosave=False, notion=''):
        """
        Trains the student.
        :param autosave: whether to save model parameters. Default is False
        :param notion: additional notes in save name
        :return: trained student
        """
        self.logger(mode='s')
        s_optimizer = self.args['s'].optimizer(self.__train_models_s__(), lr=self.args['s'].learning_rate)

        for epoch in range(self.args['s'].epochs):

            # =====================train============================
            params = self.__train_models_s__()
            LOSS_TERMS = self.plot_terms['s']['loss'].keys()
            EPOCH_LOSS = {key: [] for key in LOSS_TERMS}

            for idx, (data_x, data_y, index) in enumerate(self.train_loader, 0):
                data_x = data_x.to(torch.float32).to(self.args['s'].device)
                data_y = data_y.to(torch.float32).to(self.args['s'].device)

                PREDS = self.calculate_loss_s(data_x, data_y)
                s_optimizer.zero_grad()
                self.temp_loss['LOSS'].backward()
                s_optimizer.step()

                for key in LOSS_TERMS:
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

                if idx % (len(self.train_loader) // 5) == 0:
                    print(f"\rStudent: epoch={epoch}/{self.args['t'].epochs}, batch={idx}/{len(self.train_loader)}, "
                          f"loss={self.temp_loss['LOSS'].item()}", end='')

            for key in LOSS_TERMS:
                self.train_loss['s'][key].append(np.average(EPOCH_LOSS[key]))

            # =====================valid============================
            self.__test_models_s__()
            EPOCH_LOSS = {key: [] for key in LOSS_TERMS}

            for idx, (data_x, data_y, index) in enumerate(self.valid_loader, 0):
                data_x = data_x.to(torch.float32).to(self.args['s'].device)
                data_y = data_y.to(torch.float32).to(self.args['s'].device)
                with torch.no_grad():
                    PREDS = self.calculate_loss_s(data_x, data_y)

                for key in LOSS_TERMS:
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

            for key in LOSS_TERMS:
                self.valid_loss['s'][key].append(np.average(EPOCH_LOSS[key]))

        if autosave:
            save_path = f'../saved/{notion}/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.models['csien'].state_dict(),
                       f"{save_path}{notion}_{self.models['csien']}_{self.current_title()}.pth")

    def test_teacher(self, mode='test'):
        """
        Tests the teacher and saves estimates.
        :param mode: 'test' or 'train' (selects data loader). Default is 'test'
        :return: test results
        """
        self.test_loss['t'] = self.__gen_teacher_test__()
        self.__test_models_t__()
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
                    PREDS = self.calculate_loss_t(None, gt, ind)

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
        """
        Tests the student and saves estimates.
        :param mode: 'test' or 'train' (selects data loader). Default is 'test'
        :return: test results
        """
        self.test_loss['s'] = self.__gen_student_test__()
        self.__test_models_s__()
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
                    PREDS = self.calculate_loss_s(csi, gt, ind)

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

    def plot_train_loss(self, mode='t', double_y=False, autosave=False, notion=''):
        """
        Plots training loss.
        :param mode: 't' or 's'. Default is 't'
        :param double_y: whether to plot training loss and validation loss with double y axes. Default is False
        :param autosave: whether to save the plots. Default is False
        :param notion: additional notes in save name
        :return: plots of training loss
        """
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

    def plot_test(self, mode='t', select_ind=None, select_num=8, autosave=False, notion=''):
        """
        Plots test results.
        :param mode: 't' ot 's'. Default is 't'
        :param select_ind: specify a list of indices of samples
        :param select_num: specify the number of samples to be displayed
        :param autosave: whether to save the plots. Default is False
        :param notion: additional notes in save name
        :return: test results
        """
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
        samples = np.array(self.test_loss[mode]['IND'])[inds]
        inds = inds[np.argsort(samples)]

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
                axes[j].set_ylim(-1, 1)
                axes[j].set_title(f"#{samples[j]}")
                axes[j].grid()

            axes[0].legend()

            if autosave:
                plt.savefig(filename[mode]['LATENT'])
            plt.show()

    def traverse_latent(self, img_ind, dataset, mode='t', img='x', dim1=0, dim2=1,
                        granularity=11, autosave=False, notion=''):
        self.__plot_settings__()
        self.__test_models_s__()

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
            z = self.models['imgen'](torch.from_numpy(image).to(torch.float32).to(self.args['t'].device))
        if mode == 's':
            z = self.models['imgen'](torch.from_numpy(csi).to(torch.float32).to(self.args['s'].device))

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
                output = self.models['imgde'](torch.from_numpy(z).to(self.args['t'].device))
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
        """
        Saves all the model parameters.
        :param notion: additional notes in save name
        :return: .pth files
        """
        save_path = f'../saved/{notion}/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(self.models['imgen'].state_dict(),
                   f"{save_path}{notion}_{self.models['imgen']}{self.current_title()}.pth")
        torch.save(self.models['imgde'].state_dict(),
                   f"{save_path}{notion}_{self.models['imgde']}{self.current_title()}.pth")
        torch.save(self.models['csien'].state_dict(),
                   f"{save_path}{notion}_{self.models['csien']}{self.current_title()}.pth")

    def scheduler(self, train_t=True, train_s=True,
                  t_turns=10, s_turns=10,
                  lr_decay=False, decay_rate=0.4,
                  test_mode='train', autosave=False, notion=''):
        """
        Schedules the process of training and testing.
        :param train_t: whether to train the teacher. True or False. Default is True
        :param train_s: whether to train the student. True or False. Default is True
        :param t_turns: number of turns to run teacher train-test operations. Default is 10
        :param s_turns: number of turns to run student train-test operations. Default is 10
        :param lr_decay: whether to decay learning rate in training. Default is False
        :param decay_rate: decay rate of learning rate. Default it 0.4
        :param test_mode: 'train' or 'test' (data loader). Default is 'train'
        :param autosave: whether to save the plots. Default is False
        :param notion: additional notes in save name
        :return: trained models and test results
        """
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


class TrainerTSMask(TrainerTS):
    def __init__(self, img_encoder, img_decoder, csi_encoder, msk_decoder,
                 teacher_args, student_args,
                 train_loader, valid_loader, test_loader,
                 ):
        super(TrainerTSMask, self).__init__(img_encoder=img_encoder, img_decoder=img_decoder, csi_encoder=csi_encoder,
                                            teacher_args=teacher_args, student_args=student_args,
                                            train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
         )
        self.mask_loss = nn.BCELoss(reduction='sum')
        self.models['mskde'] = msk_decoder

    @staticmethod
    def __gen_teacher_train__():
        t_train_loss = {'learning_rate': [],
                        'epochs': [0],
                        'LOSS': [],
                        'MASK': [],
                        }

        return t_train_loss

    @staticmethod
    def __gen_teacher_test__():
        t_test_loss = {'LOSS': [],
                       'MASK': [],
                       'PRED': [],
                       'PRED_MASK': [],
                       'GT': [],
                       'GT_MASK': [],
                       'IND': []}
        return t_test_loss

    @staticmethod
    def __teacher_plot_terms__():
        terms = {'loss': {'LOSS': 'Loss',
                          'MASK': 'Mask Loss'
                          },
                 'predict': ('GT', 'GT_MASK', 'PRED', 'PRED_MASK', 'IND'),
                 'test': {'GT': 'Ground Truth',
                          'GT_MASK': 'GT Mask',
                          'PRED': 'Estimated',
                          'PRED_MASK': 'Estimated Mask'
                          }
                 }
        return terms

    def __train_models_t__(self):
        self.models['imgen'].train()
        self.models['imgde'].train()
        return [{'params': self.models['imgen'].parameters()},
                {'params': self.models['imgde'].parameters()},
                {'params': self.models['mskde'].parameters()}]

    def __test_models_t__(self):
        self.models['imgen'].eval()
        self.models['imgde'].eval()
        self.models['mskde'].eval()

    def loss(self, y, m, gt_y, gt_m, latent):
        # reduction = 'sum'
        recon_loss = self.args['t'].criterion(y, gt_y) / y.shape[0]
        mask_loss = self.mask_loss(m, gt_m) / m.shape[0]
        # loss = recon_loss + kl_loss * self.kl_weight + mask_loss
        loss = mask_loss
        return loss, recon_loss, mask_loss

    def calculate_loss_t(self, x, y, i=None):

        gt_mask = torch.where(y > 0, 1., 0.)

        latent = self.models['imgen'](y)
        output = self.models['imgde'](latent)
        mask = self.models['mskde'](latent)
        # output = output.mul(mask)
        loss, recon_loss, mask_loss = self.loss(output, mask, y, gt_mask, latent)
        self.temp_loss = {'LOSS': loss,
                          'MASK': mask_loss}
        return {'GT': y,
                'GT_MASK': gt_mask,
                'PRED': output,
                'PRED_MASK': mask,
                'IND': i}
