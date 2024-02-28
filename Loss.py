import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
from sklearn.manifold import TSNE

"""
These definitions are not for loss functions.\n
These are MyLoss classes that can store and plot losses.\n
"""


class MyLoss:
    def __init__(self, name, loss_terms, pred_terms):
        self.name = name
        self.loss = {'train': {term: [] for term in loss_terms},
                     'valid': {term: [] for term in loss_terms},
                     'test': {term: [] for term in loss_terms},
                     'pred': {term: [] for term in pred_terms}
                     }
        self.lr = []
        self.epochs = [0, 1]
        self.loss_terms = loss_terms
        self.pred_terms = pred_terms
        self.select_inds = None
        self.select_num = 8

    @staticmethod
    def __plot_settings__():
        """
        Prepares plot configurations.
        :return: plt args
        """
        # plt.style.use('default')
        # plt.rcdefaults()
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

    def logger(self, lr):

        # First round
        if not self.lr:
            self.lr.append(lr)
        else:
            # Keeping learning rate
            if lr == self.lr[-1]:
                self.epochs[-1] += 1

            # Changing learning rate
            if lr != self.lr[-1]:
                last_end = self.epochs[-1]
                self.lr.append(lr)
                self.epochs.append(last_end + 1)

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

    def reset(self, *modes):
        for mode in modes:
            if mode in ('train', 'valid', 'test'):
                self.loss[mode] = {term: [] for term in self.loss_terms}
            elif mode == 'pred':
                self.loss[mode] = {term: [] for term in self.pred_terms}

    def generate_indices(self, select_ind=None, select_num=8):
        if select_ind is not None:
            self.select_inds = select_ind
        else:
            if not self.select_inds:
                inds = np.random.choice(list(range(len(self.loss['pred']['IND']))), select_num, replace=False)
                inds = np.sort(inds)
                self.select_inds = inds
                self.select_num = select_num

    def plot_train(self, title=None, plot_terms='all', double_y=False):
        self.__plot_settings__()
        stage_color = self.colors(self.lr)
        line_color = ['b', 'orange']

        if title:
            title = f"{title} @ep{self.epochs[-1]}"
        else:
            title = f"{self.name} Training Status @ep{self.epochs[-1]}"

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

            axes[i].plot(
                         self.loss['valid'][loss],
                         line_color[1], label='Valid')
            if double_y:
                ax_r = axes[i].twinx()
            else:
                ax_r = axes[i]
            ax_r.plot(
                      self.loss['train'][loss],
                      line_color[0], label='Train')
            axes[i].set_title(loss)
            axes[i].set_xlabel('#Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].grid()
            axes[i].legend()
        plt.show()
        filename = f"{self.name}_TRAIN@ep{self.epochs[-1]}.jpg"
        return fig, filename

    def plot_test(self, title=None, plot_terms='all'):
        self.__plot_settings__()
        if title:
            title = f"{title} @ep{self.epochs[-1]}"
        else:
            title = f"{self.name} Test Loss @ep{self.epochs[-1]}"

        if plot_terms == 'all':
            plot_terms = list(self.loss['test'].keys())
        samples = np.array(self.loss['pred']['IND'])[self.select_inds]

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
            for j in range(self.select_num):
                axes[i].scatter(self.select_inds[j], self.loss['test'][item][self.select_inds[j]],
                                c='magenta', marker=(5, 1), linewidths=4)
                axes[i].annotate(str(samples[j]),
                                 (self.select_inds[j], self.loss['test'][item][self.select_inds[j]]))
        plt.show()
        filename = f"{self.name}_TEST@ep{self.epochs[-1]}.jpg"
        return fig, filename

    def plot_predict(self, plot_terms, title=None):
        self.__plot_settings__()
        if title:
            title = f"{title} @ep{self.epochs[-1]}"
        else:
            title = f"{self.name} Image Predicts @ep{self.epochs[-1]}"
        samples = np.array(self.loss['pred']['IND'])[self.select_inds]

        fig = plt.figure(constrained_layout=True)
        fig.suptitle(title)
        subfigs = fig.subfigures(nrows=len(plot_terms), ncols=1)

        for i, item in enumerate(plot_terms):
            subfigs[i].suptitle(item)
            axes = subfigs[i].subplots(nrows=1, ncols=self.select_num)
            for j in range(len(axes)):
                img = axes[j].imshow(self.loss['pred'][item][self.select_inds[j]], vmin=0, vmax=1)
                axes[j].axis('off')
                axes[j].set_title(f"#{samples[j]}")
            subfigs[i].colorbar(img, ax=axes, shrink=0.8)
        plt.show()
        filename = f"{self.name}_PRED@ep{self.epochs[-1]}.jpg"
        return fig, filename

    def plot_latent(self, plot_terms, title=None, ylim=(-1, 1)):
        self.__plot_settings__()
        if title:
            title = f"{title} @ep{self.epochs[-1]}"
        else:
            title = f"{self.name} Latent Predicts @ep{self.epochs[-1]}"
        samples = np.array(self.loss['pred']['IND'])[self.select_inds]
        colors = ('blue', 'orange')

        fig = plt.figure(constrained_layout=True)
        fig.suptitle(title)
        axes = fig.subplots(nrows=2, ncols=np.ceil(self.select_num / 2).astype(int))
        axes = axes.flatten()
        for j in range(self.select_num):
            for no, item in enumerate(plot_terms):
                axes[j].bar(range(len(self.loss['pred'][item][self.select_inds[j]])),
                            self.loss['pred'][item][self.select_inds[j]],
                            width=1, fc=colors[no], alpha=0.8, label=item)
            if ylim is not None:
                axes[j].set_ylim(*ylim)

            axes[j].set_title(f"#{samples[j]}")
            axes[j].grid()

        axes[0].legend()
        plt.show()
        filename = f"{self.name}_LAT@ep{self.epochs[-1]}.jpg"
        return fig, filename

    def plot_tsne(self, plot_terms, title=None):
        self.__plot_settings__()
        # plt.style.use('dark_background')
        if title:
            title = f"{title} @ep{self.epochs[-1]}"
        else:
            title = f"{self.name} T-SNE @ep{self.epochs[-1]}"
        samples = np.array(self.loss['pred']['IND'])[self.select_inds]
        tsne = {}

        for item in plot_terms:
            unit_shape = np.array(self.loss['pred'][item]).shape
            tsne[item] = TSNE(n_components=2, random_state=33).fit_transform(
                np.array(self.loss['pred']['GT']).reshape(unit_shape[0], -1))

        fig = plt.figure(constrained_layout=True)
        fig.suptitle(title)
        axes = fig.subplots(nrows=len(plot_terms), ncols=1)
        for i, item in enumerate(plot_terms):
            axes[i].suptitle(item)
            axes[i].scatter(tsne[item][:, 0], tsne[item][:, 1],  alpha=0.6)
            for j in range(self.select_num):
                axes[i].scatter(tsne[item][self.select_inds[j], 0], tsne[item][self.select_inds[j], 1],
                                   c='magenta', marker=(5, 1), linewidths=4)
                axes[i].annotate(str(samples[j]),
                                    (tsne[item][self.select_inds[j], 0], tsne[item][self.select_inds[j], 1]))

        plt.show()
        filename = f"{self.name}_TSNE@ep{self.epochs[-1]}.jpg"
        return fig, filename


class MyLossBBX(MyLoss):
    def __init__(self, name, loss_terms, pred_terms):
        super(MyLossBBX, self).__init__(name, loss_terms, pred_terms)

    def plot_bbx(self, title=None):
        self.__plot_settings__()
        if title:
            title = f"{title} @ep{self.epochs[-1]}"
        else:
            title = f"{self.name} Bounding Box Predicts @ep{self.epochs[-1]}"
        samples = np.array(self.loss['pred']['IND'])[self.select_inds]

        fig = plt.figure(constrained_layout=True)
        fig.suptitle(title)
        axes = fig.subplots(nrows=2, ncols=np.ceil(self.select_num / 2).astype(int))
        axes = axes.flatten()
        for j in range(self.select_num):
            axes[j].set_xlim([0, 226])
            axes[j].set_ylim([0, 128])
            x, y, w, h = self.loss['pred']['GT_BBX'][self.select_inds[j]]
            axes[j].add_patch(Rectangle((x, y), w, h, edgecolor='blue', fill=False, lw=4, label='GroundTruth'))
            x, y, w, h = self.loss['pred']['S_BBX'][self.select_inds[j]]
            axes[j].add_patch(Rectangle((x, y), w, h, edgecolor='orange', fill=False, lw=4, label='Student'))
            axes[j].axis('off')
            axes[j].set_title(f"#{samples[j]}")

        axes[0].legend()
        plt.show()
        filename = f"{self.name}_BBX@ep{self.epochs[-1]}.jpg"
        return fig, filename
