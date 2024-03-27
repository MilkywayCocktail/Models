import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random
import cv2


class ResultCalculator:
    def __init__(self, name, pred_path, gt_path, gt_ind_path):
        self.name = name

        self.pred: dict = np.load(pred_path, allow_pickle=True).item() if pred_path else None
        self.gt = np.load(gt_path) if gt_path else None
        self.gt_ind = np.load(gt_ind_path) if gt_ind_path else None
        self.image_size = (128, 226)  # in rows * columns
        self.resized = np.zeros((len(self.gt), *self.image_size))
        self.loss = F.mse_loss
        self.result = np.zeros_like(self.gt_ind)
        self.bin_edges = None
        self.hist = None

    def resize(self):
        print(f"{self.name} resizing...", end='')
        for i in range(len(self.gt)):
            if 'PRED' in self.pred.keys():
                self.resized[i] = cv2.resize(self.pred['PRED'][i], (self.image_size[1], self.image_size[0]))
            elif 'S_PRED' in self.pred.keys():
                self.resized[i] = cv2.resize(self.pred['S_PRED'][i], (self.image_size[1], self.image_size[0]))
        print("Done!")

    def calculate_loss(self):
        print(f"{self.name} calculating loss...", end='')
        for i in range(len(self.gt)):
            ind = self.pred['IND'][i]
            _ind = np.where(self.gt_ind == ind)
            pred = torch.from_numpy(self.resized[i])
            self.result[i] = F.mse_loss(pred, torch.from_numpy(self.gt[_ind]))
        print("Done")
        if np.any(np.isnan(self.result)):
            print("nan detected!")

    def calculate_hist(self):
        print(f"{self.name} calculating histograms...", end='')
        self.hist, self.bin_edges = np.histogram(self.result)
        print("Done!")


def gather_plot(*args: ResultCalculator):
    _ = plt.figure()
    mpl.rcParams['figure.figsize'] = (20, 10)
    mpl.rcParams["figure.titlesize"] = 35
    mpl.rcParams['lines.markersize'] = 10
    mpl.rcParams['axes.titlesize'] = 30
    mpl.rcParams['axes.labelsize'] = 30
    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['ytick.labelsize'] = 20
    fig = plt.figure(constrained_layout=True)
    fig.suptitle('Comparison Results')

    for ar in args:
        width = (ar.bin_edges[1] - ar.bin_edges[0]) * 0.8
        cdf = np.cumsum(ar.hist / sum(ar.hist))
        plt.bar(ar.bin_edges[1:], ar.hist / max(ar.hist), width=width, alpha=0.5, label=ar.name)
        plt.plot(ar.bin_edges[1:], cdf, '-*', label=ar.name)

    plt.ylim([0, 1])
    plt.title('Test PDF-CDF', fontweight="bold")
    plt.xlabel('Per-sample Loss')
    plt.ylabel('Frequency')
    plt.grid()
    plt.legend()
    plt.show()


def visualization(*args: ResultCalculator, select_ind=None):
    _ = plt.figure()
    mpl.rcParams['figure.figsize'] = (20, 10)
    mpl.rcParams["figure.titlesize"] = 35
    mpl.rcParams['lines.markersize'] = 10
    mpl.rcParams['axes.titlesize'] = 30
    mpl.rcParams['axes.labelsize'] = 30
    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['ytick.labelsize'] = 20
    fig = plt.figure(constrained_layout=True)
    fig.suptitle('Comparison Results')

    if not select_ind:
        select_ind = np.random.choice(args[0].gt_ind, 8, replace=False).astype(int)

    subfigs = fig.subfigures(nrows=len(args) + 1, ncols=1)

    subfigs[0].suptitle("Ground Truth", fontweight="bold")
    axes = subfigs[0].subplots(nrows=1, ncols=8)
    for j in range(len(axes)):
        ind = args[0].pred['IND'][j]
        _ind = np.where(args[0].gt_ind == ind)
        img = axes[j].imshow(args[0].gt[_ind], vmin=0, vmax=1)
        axes[j].axis('off')
        axes[j].set_title(f"#{_ind}")

    for i, ar in enumerate(args):
        subfigs[i+1].suptitle(ar.name, fontweight="bold")
        axes = subfigs[i+1].subplots(nrows=1, ncols=8)
        for j in range(len(axes)):
            ind = ar.pred['IND'][j]
            _ind = np.where(args[0].gt_ind == ind)
            img = axes[j].imshow(ar.resized[select_ind[j]], vmin=0, vmax=1)
            axes[j].axis('off')
            axes[j].set_title(f"#{_ind}")
    plt.show()

