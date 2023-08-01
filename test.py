import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from torchvision import datasets

y = [1,2,3,4,5,6]
y2 = [4,5,7,6,7,1]
l = list(range(len(y)))

st = 7.89
s = np.ceil(st).astype(int)
print(s)


def plot_loss(self, mode, double_y, notion, autosave):
    self.__plot_settings__()
    # Training & Validation Loss
    if mode == 't':
        title = f"Teacher Training Status @ep{self.train_loss['t']['epochs'][-1]}"
        savename = f"{self.current_title()}_T_train_{notion}.jpg"
        contents = self.train_loss['t']
        plot_items = self.plot_terms['t']['train']
    elif mode == 's':
        title = f"Student Training Status @ep{self.train_loss['s']['epochs'][-1]}"
        savename = f"{self.current_title()}_S_train_{notion}.jpg"
        contents = self.train_loss['s']
        plot_items = self.plot_terms['s']['train']

    stage_color = self.colors(contents['learning_rate'])
    line_color = ['b', 'orange']
    fig = plt.figure(constrained_layout=True)
    fig.suptitle(title)
    if len(plot_items.keys()) == 1:
        axes = [plt.gca()]
    elif len(plot_items.keys()) > 3:
        axes = fig.subplots(nrows=2, ncols=np.ceil(len(plot_items.keys()) / 2).astype(int))
        axes = axes.flatten()
    else:
        axes = fig.subplots(1, len(plot_items.keys()))
        axes = axes.flatten()

    for i, loss in enumerate(plot_items.keys()):
        for j, learning_rate in enumerate(contents['learning_rate']):
            axes[i].axvline(contents['epochs'][j],
                            linestyle='--',
                            color=stage_color[j],
                            label=f'lr={learning_rate}')

        axes[i].plot(list(range(len(contents[plot_items[loss][1]]))),
                     contents[plot_items[loss][1]],
                     line_color[1], label=plot_items[loss][1])
        if double_y:
            ax_r = axes[i].twinx()
        else:
            ax_r = axes[i]
        ax_r.plot(list(range(len(contents[plot_items[loss][0]]))),
                  contents[plot_items[loss][0]],
                  line_color[0], label=plot_items[loss][0])
        axes[i].set_title(loss)
        axes[i].set_xlabel('#Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].grid()
        axes[i].legend()

    if autosave:
        plt.savefig(savename)
    plt.show()
