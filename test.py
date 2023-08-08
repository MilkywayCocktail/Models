import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from torchvision import datasets

y = [1,2,3,4,5,6]
y2 = [4,5,7,6,7,1]
l = list(range(len(y)))

keys = ['A', 'B', 'C']

dics = {key:[] for key in keys}
print(dics.items())

dic2 = {'A':2, 'B':3, 'C':4}
dics = {**dics, **dic2}

print(f"\r123445", end='')
print(f"\raabbc")


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