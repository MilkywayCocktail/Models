import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib import cm
from sklearn.manifold import TSNE
import os

"""
These definitions are not for loss functions.\n
These are MyLoss classes that can store and plot losses.\n
"""


class MyLoss:
    def __init__(self, name, loss_terms, pred_terms, dataset: str = 'TRAIN'):
        self.name = name
        self.dataset = dataset
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
        self.ind_range = 0

    @staticmethod
    def __plot_settings__():
        """
        Prepares plot configurations.
        :return: plt args
        """
        _ = plt.figure()
        mpl.rcParams['figure.figsize'] = (20, 10)
        mpl.rcParams["figure.titlesize"] = 35
        mpl.rcParams['lines.markersize'] = 10
        mpl.rcParams['axes.titlesize'] = 30
        mpl.rcParams['axes.labelsize'] = 30
        mpl.rcParams['xtick.labelsize'] = 20
        mpl.rcParams['ytick.labelsize'] = 20
        fig = plt.figure(constrained_layout=True)
        return fig

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
        # for i in range(len(self.loss['pred']['IND'])):
        #     self.loss['pred']['IND'][i] = self.loss['pred']['IND'][i].astype(int).tolist()

    def reset(self, *modes, dataset: str = 'TRAIN'):
        for mode in modes:
            if mode in ('train', 'valid', 'test'):
                self.loss[mode] = {term: [] for term in self.loss_terms}
            elif mode == 'pred':
                self.loss[mode] = {term: [] for term in self.pred_terms}
        self.dataset = dataset.upper()

    def save(self, save_term: str = 'test', notion=None):
        save_path = f"../saved/{notion}/"
        print(f"Saving {save_term} including {', '.join([key for key in self.loss[save_term].keys()])}...", end='')
        np.save(f"{save_path}{notion}_{self.name}_{save_term}.npy", self.loss[save_term])
        print('Done')

    def generate_indices(self, select_ind: list = None, select_num=8):
        if select_ind:
            self.select_inds = np.array(select_ind)
        else:
            if not np.any(self.select_inds) or self.ind_range != len(self.loss['pred']['IND']):
                self.ind_range = len(self.loss['pred']['IND'])
                inds = np.random.choice(np.arange(self.ind_range), select_num, replace=False).astype(int)
                inds = np.sort(inds)
                self.select_inds = inds
                self.select_num = select_num

    def plot_train(self, title=None, plot_terms='all', double_y=False):
        stage_color = self.colors(self.lr)
        line_color = ['b', 'orange']

        if title:
            title = f"{title} @ep{self.epochs[-1]}"
        else:
            title = f"{self.name} Training Status @ep{self.epochs[-1]}"

        if plot_terms == 'all':
            plot_terms = list(self.loss['train'].keys())

        fig = self.__plot_settings__()
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
            axes[i].set_title(loss, fontweight="bold")
            axes[i].set_xlabel('#Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].grid()
            axes[i].legend()
        plt.show()
        filename = f"{self.name}_TRAIN@ep{self.epochs[-1]}.jpg"
        return fig, filename

    def plot_test(self, title=None, plot_terms='all'):
        if title:
            title = f"{title} @ep{self.epochs[-1]}"
        else:
            title = f"{self.name} Test Loss on {self.dataset} @ep{self.epochs[-1]}"

        if plot_terms == 'all':
            plot_terms = list(self.loss['test'].keys())

        fig = self.__plot_settings__()
        fig.suptitle(title)
        plt.yscale('log', base=2)
        plt.ylim([2 ** -18, 2])
        plt.axhline(1, linestyle='--', color='lightgreen', label='1.0')
        for i, item in enumerate(plot_terms):
            plt.boxplot([self.loss['test'][item]], labels=[item], positions=[i+1], vert=True, showmeans=True,
                        patch_artist=True,
                        boxprops={'facecolor': 'lightgreen'})

        plt.show()
        filename = f"{self.name}_TEST_{self.dataset}SET@ep{self.epochs[-1]}.jpg"
        return fig, filename

    def plot_test_cdf(self, title=None, plot_terms='all'):
        if title:
            title = f"{title} @ep{self.epochs[-1]}"
        else:
            title = f"{self.name} Test PDF-CDF on {self.dataset} @ep{self.epochs[-1]}"

        if plot_terms == 'all':
            plot_terms = list(self.loss['test'].keys())

        fig = self.__plot_settings__()
        fig.suptitle(title)
        if len(plot_terms) == 1:
            axes = [plt.gca()]
        elif len(plot_terms) > 3:
            axes = fig.subplots(2, np.ceil(len(plot_terms) / 2).astype(int))
            axes = axes.flatten()
        else:
            axes = fig.subplots(1, len(plot_terms))
            axes = axes.flatten()

        for i, item in enumerate(plot_terms):
            hist, bin_edges = np.histogram(self.loss['test'][item])
            width = (bin_edges[1] - bin_edges[0]) * 0.8
            cdf = np.cumsum(hist / sum(hist))

            axes[i].bar(bin_edges[1:], hist / max(hist), width=width, color='blue')
            axes[i].plot(bin_edges[1:], cdf, '-*', color='orange')
            axes[i].set_ylim([0, 1])
            axes[i].set_title(item, fontweight="bold")
            axes[i].set_xlabel('Per-sample Loss')
            axes[i].set_ylabel('Frequency')
            axes[i].grid()

        plt.show()
        filename = f"{self.name}_PDF_{self.dataset}SET@ep{self.epochs[-1]}.jpg"
        return fig, filename

    def plot_predict(self, plot_terms, title=None):
        if title:
            title = f"{title} @ep{self.epochs[-1]}"
        else:
            title = f"{self.name} Image Predicts on {self.dataset} @ep{self.epochs[-1]}"
        samples = np.array(self.loss['pred']['IND']).astype(int)[self.select_inds]

        fig = self.__plot_settings__()
        fig.suptitle(title)
        subfigs = fig.subfigures(nrows=len(plot_terms), ncols=1)

        for i, item in enumerate(plot_terms):
            subfigs[i].suptitle(item)
            axes = subfigs[i].subplots(nrows=1, ncols=self.select_num)
            for j in range(len(axes)):
                img = axes[j].imshow(self.loss['pred'][item][self.select_inds[j]], vmin=0, vmax=1)
                axes[j].axis('off')
                axes[j].set_title(f"#{samples[j]}", fontweight="bold")
            subfigs[i].colorbar(img, ax=axes, shrink=0.8)
        plt.show()
        filename = f"{self.name}_PRED_{self.dataset}SET@ep{self.epochs[-1]}.jpg"
        return fig, filename

    def plot_latent(self, plot_terms, title=None, ylim: tuple = (-1, 1)):
        if title:
            title = f"{title} @ep{self.epochs[-1]}"
        else:
            title = f"{self.name} Latent Predicts on {self.dataset} @ep{self.epochs[-1]}"
        samples = np.array(self.loss['pred']['IND']).astype(int)[self.select_inds]
        colors = ('blue', 'orange')

        fig = self.__plot_settings__()
        fig.suptitle(title)
        axes = fig.subplots(nrows=2, ncols=np.ceil(self.select_num / 2).astype(int))
        axes = axes.flatten()
        for j in range(self.select_num):
            for no, item in enumerate(plot_terms):
                axes[j].bar(range(len(self.loss['pred'][item][self.select_inds[j]])),
                            self.loss['pred'][item][self.select_inds[j]],
                            width=1, fc=colors[no], alpha=0.8, label=item)
            if ylim:
                axes[j].set_ylim(*ylim)

            axes[j].set_title(f"#{samples[j]}", fontweight="bold")
            axes[j].grid()

        axes[0].legend()
        plt.show()
        filename = f"{self.name}_LAT_{self.dataset}SET@ep{self.epochs[-1]}.jpg"
        return fig, filename

    def plot_tsne(self, plot_terms, title=None):
        # plt.style.use('dark_background')
        if title:
            title = f"{title} @ep{self.epochs[-1]}"
        else:
            title = f"{self.name} T-SNE on {self.dataset} @ep{self.epochs[-1]}"
        samples = np.array(self.loss['pred']['IND']).astype(int)[self.select_inds]
        tsne = {}

        for item in plot_terms:
            unit_shape = np.array(self.loss['pred'][item]).shape
            tsne[item] = TSNE(n_components=2, random_state=33).fit_transform(
                np.array(self.loss['pred'][item]).reshape(unit_shape[0], -1))

        fig = self.__plot_settings__()
        fig.suptitle(title)
        axes = fig.subplots(nrows=1, ncols=len(plot_terms))
        for i, item in enumerate(plot_terms):
            axes[i].set_title(item, fontweight="bold")
            axes[i].scatter(tsne[item][:, 0], tsne[item][:, 1],  alpha=0.6)
            for j in range(self.select_num):
                axes[i].scatter(tsne[item][self.select_inds[j], 0], tsne[item][self.select_inds[j], 1],
                                c='magenta', marker=(5, 1), linewidths=4)
                axes[i].annotate(str(samples[j]),
                                 (tsne[item][self.select_inds[j], 0], tsne[item][self.select_inds[j], 1]),
                                 fontsize=20)

        plt.show()
        filename = f"{self.name}_TSNE_{self.dataset}SET@ep{self.epochs[-1]}.jpg"
        return fig, filename


class MyLossBBX(MyLoss):
    def __init__(self, depth=False, *args, **kwargs):
        super(MyLossBBX, self).__init__(*args, **kwargs)
        self.depth = depth

    def plot_bbx(self, title=None):
        if title:
            title = f"{title} @ep{self.epochs[-1]}"
        else:
            title = f"{self.name} Bounding Box Predicts on {self.dataset} @ep{self.epochs[-1]}"
        samples = np.array(self.loss['pred']['IND']).astype(int)[self.select_inds]

        fig = self.__plot_settings__()
        fig.suptitle(title)
        axes = fig.subplots(nrows=2, ncols=np.ceil(self.select_num / 2).astype(int))
        axes = axes.flatten()
        for j in range(self.select_num):
            axes[j].set_xlim([0, 226])
            axes[j].set_ylim([0, 128])
            axes[j].set_title(f"#{samples[j]}", fontweight="bold")
            x, y, w, h = self.loss['pred']['GT_BBX'][self.select_inds[j]]
            x = int(x * 226)
            y = int(y * 128)
            w = int(w * 226)
            h = int(h * 128)

            axes[j].add_patch(Rectangle((x, y), w, h, edgecolor='blue', fill=False, lw=4, label='GroundTruth'))
            if self.depth:
                axes[j].annotate(f"GT={self.loss['pred']['GT_DPT'][self.select_inds[j]]:.2f}",
                                 (48, 20),
                                 fontsize=20, color='blue')
            x, y, w, h = self.loss['pred']['S_BBX'][self.select_inds[j]]
            axes[j].add_patch(Rectangle((x, y), w, h, edgecolor='orange', fill=False, lw=4, label='Student'))
            if self.depth:
                axes[j].annotate(f"Pred={self.loss['pred']['S_DPT'][self.select_inds[j]]:.2f}",
                                 (48, 10),
                                 fontsize=20, color='orange')
                axes[j].scatter(48, 20,
                                c='blue', marker=(5, 1), linewidths=4)
                axes[j].scatter(48, 10,
                                c='orange', marker=(5, 1), linewidths=4)
            axes[j].axis('off')
            axes[j].add_patch(plt.Rectangle((0, 0), 226, 128, facecolor="#eafff5",
                                            transform=axes[j].transAxes, zorder=-1))

        axes[0].legend()
        plt.show()
        filename = f"{self.name}_BBX_{self.dataset}SET@ep{self.epochs[-1]}.jpg"
        return fig, filename

