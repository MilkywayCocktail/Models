import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2
from misc import plot_settings


class ResultCalculator:
    def __init__(self, name, pred_path, gt=None, gt_ind=None):
        self.name = name

        self.preds: dict = np.load(pred_path, allow_pickle=True).item() if pred_path else None
        self.gt = gt
        self.gt_ind = gt_ind
        self.image_size = (128, 226)  # in rows * columns
        self.resized = np.zeros((len(self.gt), *self.image_size))
        self.loss = F.mse_loss
        self.result = np.zeros_like(self.gt_ind)
        self.bin_edges = None
        self.hist = None

    def resize(self):
        print(f"{self.name} resizing...", end='')
        for i in range(len(self.preds['IND'])):
            if 'PRED' in self.preds.keys():
                self.resized[i] = cv2.resize(self.preds['PRED'][i], (self.image_size[1], self.image_size[0]))
            elif 'S_PRED' in self.preds.keys():
                self.resized[i] = cv2.resize(self.preds['S_PRED'][i], (self.image_size[1], self.image_size[0]))
        print("Done!")

    def calculate_loss(self):
        print(f"{self.name} calculating loss...", end='')
        for i in range(len(self.preds['IND'])):
            ind = self.preds['IND'][i]
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


class PropResultCalculator(ResultCalculator):
    def __init__(self, *args, **kwargs):
        super(PropResultCalculator, self).__init__(*args, **kwargs)

        self.bbx = np.array(self.preds['S_BBX'])
        self.depth = np.array(self.preds['S_DPT'])
        self.mask = np.array(self.preds['S_PRED'])
        self.inds = self.preds['IND']

        print(f"Loaded bbx of {self.bbx.shape}, depth of {self.depth.shape}, mask of {self.mask.shape}")

        self.imgs = np.zeros((len(self.bbx), *self.image_size))
        self.min_area = 0
        self.fail_count = 0

    def reconstruct(self):
        print("Reconstructing...", end='')
        for i in range(len(self.inds)):
            img = np.squeeze(self.mask[i]).astype('float32')
            (T, timg) = cv2.threshold((img * 255).astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(timg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) != 0:
                contour = max(contours, key=lambda x: cv2.contourArea(x))
                area = cv2.contourArea(contour)

                if area < self.min_area:
                    # print(area)
                    pass

                else:
                    x, y, w, h = cv2.boundingRect(contour)
                    subject = img[y:y + h, x:x + w] * np.squeeze(self.depth[i])

                    x1, y1, x2, y2 = self.bbx[i]
                    x1 = int(x1 * 226)
                    y1 = int(y1 * 128)
                    x2 = int(x2 * 226)
                    y2 = int(y2 * 128)
                    w1 = x2 - x1
                    h1 = y2 - y1

                    try:
                        subject1 = cv2.resize(subject, (w1, h1))
                        for x in range(w1):
                            for y in range(h1):
                                self.imgs[i, y1 + y, x1 + x] = subject1[y, x]
                    except Exception as e:
                        print(e)
                        print(x1, y1, x2, y2, w1, h1)
                        print(subject1.shape)
                        self.fail_count += 1
        print("Done")
        print(f"Reconstruction finished. Failure count = {self.fail_count}")

    def plot_example(self, inds=None):
        fig = plot_settings()
        fig.suptitle('Reconstruction Examples')

        subfigs = fig.subfigures(nrows=4, ncols=1)

        plot_terms = {'Cropped Ground Truth': self.preds['GT'],
                      'Cropped Estimates': self.preds['S_PRED'],
                      'Raw Ground Truth': self.gt,
                      'Raw Estimates': self.resized}

        if not inds:
            inds = np.random.choice(np.arange(len(self.preds['IND'])), 8, replace=False).astype(int)
            inds = np.sort(inds)

        for i, (key, value) in enumerate(plot_terms.items()):
            subfigs[i].suptitle(key, fontweight="bold")
            axes = subfigs[i].subplots(nrows=1, ncols=8)
            for j in range(len(axes)):
                ind = self.preds['IND'][inds[j]]
                _ind = np.where(self.gt_ind == ind)
                img = axes[j].imshow(value[_ind] if key == 'Raw Ground Truth' else value[inds[j]], vmin=0, vmax=1)
                if key == 'Raw Ground Truth':
                    x, y, w, h = self.preds['GT_BBX'][inds[j]]
                    x = int(x * 226)
                    y = int(y * 128)
                    w = int(w * 226)
                    h = int(h * 128)
                    axes[j].add_patch(Rectangle((x, y), w, h, edgecolor='blue', fill=False, lw=3))
                elif key == 'Raw Estimates':
                    x, y, w, h = self.preds['S_BBX'][inds[j]]
                    x = int(x * 226)
                    y = int(y * 128)
                    w = int(w * 226)
                    h = int(h * 128)
                    axes[j].add_patch(Rectangle((x, y), w, h, edgecolor='orange', fill=False, lw=3))
                axes[j].axis('off')
                axes[j].set_title(f"#{_ind}")
        plt.show()
        filename = f"{self.name}_Reconstruct.jpg"
        return fig, filename


def gather_plot(*args: ResultCalculator):
    fig = plot_settings()
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
    filename = f"comparison_CDF.jpg"
    return fig, filename


def visualization(*args: ResultCalculator, inds=None):
    fig = plot_settings()
    fig.suptitle('Comparison Visualization')

    if not inds:
        inds = np.random.choice(np.arange(len(args[0].preds['IND'])), 8, replace=False).astype(int)
        inds = np.sort(inds)

    subfigs = fig.subfigures(nrows=len(args) + 1, ncols=1)

    subfigs[0].suptitle("Ground Truth", fontweight="bold")
    axes = subfigs[0].subplots(nrows=1, ncols=8)
    for j in range(len(axes)):
        ind = args[0].preds['IND'][inds[j]]
        _ind = np.where(args[0].gt_ind == ind)
        img = axes[j].imshow(args[0].gt[_ind], vmin=0, vmax=1)
        axes[j].axis('off')
        axes[j].set_title(f"#{_ind}")

    for i, ar in enumerate(args):
        subfigs[i+1].suptitle(ar.name, fontweight="bold")
        axes = subfigs[i+1].subplots(nrows=1, ncols=8)
        for j in range(len(axes)):
            ind = ar.preds['IND'][inds[j]]
            _ind = np.where(args[0].gt_ind == ind)
            img = axes[j].imshow(ar.resized[inds[j]], vmin=0, vmax=1)
            axes[j].axis('off')
            axes[j].set_title(f"#{_ind}")
    plt.show()
    filename = f"comparison_visual.jpg"
    return fig, filename

