import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm


class MyLoss:
    def __init__(self, loss_terms, pred_terms):
        self.loss = {'train': {term: [] for term in loss_terms},
                     'valid': {term: [] for term in loss_terms},
                     'test': {term: [] for term in loss_terms},
                     'pred': {term: [] for term in pred_terms}
                     }
        self.lr = []
        self.epochs = [0]
        self.loss_terms = loss_terms
        self.pred_terms = pred_terms

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
    def colors(arrays):
        """
        Color solution for plotting loss
        :param arrays: array of learning rates
        :return: variation of colors
        """
        arr = -np.log(arrays)
        norm = plt.Normalize(arr.min(), arr.max())
        map_vir = cm.get_cmap(name='viridis')
        c = map_vir(norm(arr))
        return c

    def logger(self, lr, epochs):

        # First round
        if not self.lr:
            self.lr.append(lr)
            self.epochs.append(epochs)
        else:
            # Not changing learning rate
            if lr == self.lr[-1]:
                self.epochs[-1] += epochs

            # Changing learning rate
            if lr != self.lr[-1]:
                last_end = self.epochs[-1]
                self.lr.append(lr)
                self.epochs.append(last_end + epochs)

    def update(self, mode, losses):
        if mode in ('train', 'valid'):
            for key in losses.keys():
                self.loss[mode][key].append(losses[key].squeeze())
        elif mode == 'test':
            for key in losses.keys():
                self.loss[mode][key] = losses[key]
        elif mode == 'pred':
            for key in losses.keys():
                self.loss[mode][key].append(losses[key].cpu().detach().numpy().squeeze())

    def reset(self, mode):
        if mode in ('train', 'valid', 'test'):
            self.loss[mode] = {term: [] for term in self.loss_terms}
        elif mode == 'pred':
            self.loss[mode] = {term: [] for term in self.pred_terms}

    def plot_train(self, title, plot_terms='all', double_y=False):
        self.__plot_settings__()
        stage_color = self.colors(self.lr)
        line_color = ['b', 'orange']

        title = f"{title} @ep{self.epochs[-1]}"

        if plot_terms == 'all':
            plot_terms = list(self.loss['train'].keys())

        fig = plt.figure(constrained_layout=True)
        fig.suptitle(title)

        if len(plot_terms) == 1:
            axes = [plt.gca()]
        elif len(plot_terms) > 3:
            axes = fig.subplots(2, np.ceil(len(plot_terms)/2).astype(int))
            axes = axes.flatten()
        else:
            axes = fig.subplots(1, len(plot_terms))
            axes = axes.flatten()

        for i, loss in enumerate(plot_terms):
            for j, learning_rate in enumerate(self.lr):
                axes[i].axvline(self.epochs[j],
                                linestyle='--',
                                color=stage_color[j],
                                label=f'lr={learning_rate}')

            axes[i].plot(list(range(len(self.loss['valid'][loss]))),
                         self.loss['valid'][loss],
                         line_color[1], label='Valid')
            if double_y:
                ax_r = axes[i].twinx()
            else:
                ax_r = axes[i]
            ax_r.plot(list(range(len(self.loss['train'][loss]))),
                      self.loss['train'][loss],
                      line_color[0], label='Train')
            axes[i].set_title(loss)
            axes[i].set_xlabel('#Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].grid()
            axes[i].legend()
        plt.show()

    def plot_test(self, title, select_ind, plot_terms='all'):
        self.__plot_settings__()

        title = f"{title} @ep{self.epochs[-1]}"

        if plot_terms == 'all':
            plot_terms = list(self.loss['test'].keys())

        fig = plt.figure(constrained_layout=True)
        fig.suptitle(title)
        if len(plot_terms) == 1:
            axes = [plt.gca()]
        elif len(plot_terms) > 3:
            axes = fig.subplots(2, np.ceil(len(plot_terms)/2).astype(int))
            axes = axes.flatten()
        else:
            axes = fig.subplots(1, len(plot_terms))
            axes = axes.flatten()

        for i, item in enumerate(plot_terms):
            axes[i].scatter(list(range(len(self.loss['test'][item]))),
                            self.loss['test'][item], alpha=0.6)
            axes[i].set_title(item)
            axes[i].set_xlabel('#Sample')
            axes[i].set_ylabel('Loss')
            axes[i].grid()
            for j in range(len(select_ind)):
                axes[i].scatter(select_ind[j], self.loss['test'][item][select_ind[j]],
                                c='magenta', marker=(5, 1), linewidths=4)
        plt.show()

    def plot_predict(self, title, select_ind, plot_terms):
        self.__plot_settings__()

        title = f"{title} @ep{self.epochs[-1]}"
        samples = np.array(self.loss['pred']['IND'])[select_ind]

        fig = plt.figure(constrained_layout=True)
        fig.suptitle(title)
        subfigs = fig.subfigures(nrows=len(plot_terms), ncols=1)

        for i, item in enumerate(plot_terms):
            subfigs[i].suptitle(item)
            axes = subfigs[i].subplots(nrows=1, ncols=len(select_ind))
            for j in range(len(axes)):
                img = axes[j].imshow(self.loss['pred'][item][select_ind[j]], vmin=0, vmax=1)
                axes[j].axis('off')
                axes[j].set_title(f"#{samples[j]}")
            subfigs[i].colorbar(img, ax=axes, shrink=0.8)
        plt.show()


class MyLoss_S(MyLoss):
    def __init__(self, loss_terms, pred_terms):
        super(MyLoss_S, self).__init__(loss_terms, pred_terms)

    def plot_latent(self, title, select_ind):
        self.__plot_settings__()

        title = f"{title} @ep{self.epochs[-1]}"
        samples = np.array(self.loss['pred']['IND'])[select_ind]

        fig = plt.figure(constrained_layout=True)
        fig.suptitle(title)
        axes = fig.subplots(nrows=2, ncols=np.ceil(len(select_ind) / 2).astype(int))
        axes = axes.flatten()
        for j in range(len(select_ind)):
            axes[j].bar(range(len(self.loss['pred']['T_LATENT'][select_ind[0]])),
                        self.loss['pred']['T_LATENT'][select_ind[j]],
                        width=1, fc='blue', alpha=0.8, label='Teacher')
            axes[j].bar(range(len(self.loss['pred']['S_LATENT'][select_ind[0]])),
                        self.loss['pred']['S_LATENT'][select_ind[j]],
                        width=1, fc='orange', alpha=0.8, label='Student')
            axes[j].set_ylim(-1, 1)
            axes[j].set_title(f"#{samples[j]}")
            axes[j].grid()

        axes[0].legend()
        plt.show()


class MyLoss_T_BBX(MyLoss):
    def __init__(self, loss_terms, pred_terms):
        super(MyLoss_T_BBX, self).__init__(loss_terms, pred_terms)

    def plot_bbx(self, title, select_ind):
        self.__plot_settings__()

        title = f"{title} @ep{self.epochs[-1]}"
        samples = np.array(self.loss['pred']['IND'])[select_ind]

        fig = plt.figure(constrained_layout=True)
        fig.suptitle(title)
        axes = fig.subplots(nrows=2, ncols=np.ceil(len(select_ind) / 2).astype(int))
        axes = axes.flatten()
        for j in range(len(select_ind)):
            axes[j].plot([0, 128], [0, 128])
            x, y, w, h = self.loss['pred']['GT_BBX'][select_ind]
            axes[j].add_patch(Rectangle((x, y), w, h, edgecolor='red', fill=False, lw=4, label='GroundTruth'))
            x, y, w, h = self.loss['pred']['PRED_BBX'][select_ind]
            axes[j].add_patch(Rectangle((x, y), w, h, edgecolor='blue', fill=False, lw=4, label='Teacher'))
            axes[j].axis('off')
            axes[j].set_title(f"#{samples[j]}")

        axes[0].legend()
        plt.show()


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
            axes.plot([0, 128], [0, 128])
            x, y, w, h = self.loss['pred']['GT_BBX'][select_ind]
            axes[j].add_patch(Rectangle((x, y), w, h, edgecolor='red', fill=False, lw=4, label='GroundTruth'))
            x, y, w, h = self.loss['pred']['T_BBX'][select_ind]
            axes[j].add_patch(Rectangle((x, y), w, h, edgecolor='blue', fill=False, lw=4, label='Teacher'))
            x, y, w, h = self.loss['pred']['S_BBX'][select_ind]
            axes[j].add_patch(Rectangle((x, y), w, h, edgecolor='orange', fill=False, lw=4, label='Student'))
            axes[j].axis('off')
            axes[j].set_title(f"#{samples[j]}")

        axes[0].legend()
        plt.show()
