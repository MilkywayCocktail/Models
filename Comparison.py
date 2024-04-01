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
        print(f"{self.name} loading...")
        self.preds: dict = np.load(pred_path, allow_pickle=True).item() if pred_path else None
        self.inds = self.preds['IND']
        print("{name} loaded Estimates of {pred_img.shape} as {pred_img.dtype}".format(
            name=self.name,
            pred_img=np.array(self.preds['S_PRED'] if 'S_PRED' in self.preds.keys() else self.preds['PRED']))
        )
        self.gt = gt
        self.gt_ind = gt_ind
        self.image_size = (128, 226)  # in rows * columns
        self.resized = np.zeros((len(self.preds['IND']), *self.image_size))
        self.loss = F.mse_loss
        self.result = np.zeros(len(self.preds['IND']))

    def resize(self):
        print(f"{self.name} resizing...", end='')
        for i in range(len(self.preds['IND'])):
            self.resized[i] = cv2.resize(
                np.squeeze(self.preds['S_PRED'][i] if 'S_PRED' in self.preds.keys() else self.preds['PRED'][i]),
                (self.image_size[1], self.image_size[0]))
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

    def calculate_cdf(self):
        print(f"{self.name} calculating histograms...", end='')
        hist, bin_edges = np.histogram(self.result)
        print("Done!")
        fig = plot_settings()
        fig.suptitle(f"{self.name} Loss on Raw Images")
        width = (bin_edges[1] - bin_edges[0]) * 0.8
        cdf = np.cumsum(hist / sum(hist))

        plt.bar(bin_edges[1:], hist / max(hist), width=width, alpha=0.5)
        plt.plot(bin_edges[1:], cdf, '-*')

        plt.ylim([0, 1])
        plt.title('Test PDF-CDF', fontweight="bold")
        plt.xlabel('Per-sample Loss')
        plt.ylabel('Frequency')
        plt.grid()
        plt.legend()
        plt.show()
        filename = f"{self.name}_CDF.jpg"
        return fig, filename


class PropResultCalculator(ResultCalculator):
    def __init__(self, *args, **kwargs):
        super(PropResultCalculator, self).__init__(*args, **kwargs)

        self.bbx = np.array(self.preds['S_BBX'])
        self.depth = np.array(self.preds['S_DPT'])

        self.min_area = 0
        self.fail_count = 0
        self.fail_ind = []

    def resize(self):
        print("Reconstructing...", end='')
        for i in range(len(self.inds)):
            img = np.squeeze(self.preds['S_PRED'][i]) * np.squeeze(self.depth[i])
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
                    subject = img[y:y + h, x:x + w]

                    x1, y1, x2, y2 = self.bbx[i]
                    x0 = int(x1 * 226)
                    y0 = int(y1 * 128)
                    w0 = int((x2 - x1) * 226)
                    h0 = int((y2 - y1) * 128)

                    # In case of reversing (x1, y1) and (x2, y2)
                    if w0 < 0 and h0 < 0:
                        x0 = int(x2 * 226)
                        y0 = int(y2 * 128)
                        w0 = int((x1 - x2) * 226)
                        h0 = int((y1 - y2) * 128)

                    try:
                        subject1 = cv2.resize(subject, (w0, h0))
                        for x_ in range(w0):
                            for y_ in range(h0):
                                self.resized[i, y0 + y_, x0 + x_] = subject1[y_, x_]
                    except Exception as e:
                        print(e)
                        print(x0, y0,  w0, h0)
                        self.fail_count += 1
                        self.fail_ind.append(i)
        print("Done")
        print(f"Reconstruction finished. Failure count = {self.fail_count}")

    def plot_example(self, inds=None, title=None):
        fig = plot_settings()
        fig.suptitle('Reconstruction Examples' if not title else title)

        subfigs = fig.subfigures(nrows=4, ncols=1)

        plot_terms = {'Cropped Ground Truth': self.preds['GT'],
                      'Cropped Estimates': self.preds['S_PRED'],
                      'Raw Ground Truth': self.gt,
                      'Raw Estimates': self.resized}

        if not inds:
            inds = np.random.choice(np.arange(len(self.preds['IND'])), 8, replace=False).astype(int)
            inds = np.sort(inds)
        samples = np.array(self.preds['IND']).astype(int)[inds]

        for i, (key, value) in enumerate(plot_terms.items()):
            subfigs[i].suptitle(key, fontweight="bold")
            axes = subfigs[i].subplots(nrows=1, ncols=8)
            for j in range(len(axes)):
                _ind = np.where(self.gt_ind == samples[j])
                img = axes[j].imshow(np.squeeze(value[_ind]) if key == 'Raw Ground Truth'
                                     else np.squeeze(value[inds[j]]), vmin=0, vmax=1)
                if key == 'Raw Ground Truth':
                    x1, y1, x2, y2 = self.preds['GT_BBX'][inds[j]]
                    x = int(x1 * 226)
                    y = int(y1 * 128)
                    w = int((x2 - x1) * 226)
                    h = int((y2 - y1) * 128)
                    axes[j].add_patch(Rectangle((x, y), w, h, edgecolor='pink', fill=False, lw=3))
                elif key == 'Raw Estimates':
                    x1, y1, x2, y2 = self.preds['S_BBX'][inds[j]]
                    x = int(x1 * 226)
                    y = int(y1 * 128)
                    w = int((x2 - x1) * 226)
                    h = int((y2 - y1) * 128)
                    axes[j].add_patch(Rectangle((x, y), w, h, edgecolor='orange', fill=False, lw=3))
                axes[j].axis('off')
                axes[j].set_title(f"#{samples[j]}")
        plt.show()
        filename = f"{self.name}_Reconstruct.jpg"
        return fig, filename


def gather_plot(*args: ResultCalculator, title=None):
    fig = plot_settings()
    fig.suptitle('Comparison Results' if not title else title)

    bins = np.linspace(np.min([np.min(ar.result) for ar in args]), np.max([np.max(ar.result) for ar in args]), 50)

    for ar in args:
        hist_, bin_edges = np.histogram(ar.result, bins)
        width = (bin_edges[1] - bin_edges[0]) * 0.8
        cdf = np.cumsum(hist_ / sum(hist_))
        plt.bar(bin_edges[1:], hist_ / max(hist_), width=width, label=ar.name)
        plt.plot(bin_edges[1:], cdf, '-*', label=ar.name)

    ax = plt.gca()
    ax.fill_between(np.arange(0, 0.2, 0.01), 1.02, color='white', alpha=0.5, zorder=1)
    plt.title('Test PDF-CDF', fontweight="bold")
    plt.xlabel('Per-sample Loss')
    plt.ylabel('Frequency')
    plt.grid()
    plt.legend()
    plt.show()
    filename = f"comparison_CDF.jpg"

    return fig, filename


def visualization(*args: ResultCalculator, inds=None, figsize=(20, 10), title=None):
    fig = plot_settings(figsize)
    fig.suptitle('Comparison Visualization' if not title else title)

    if not inds:
        inds = np.random.choice(np.arange(len(args[0].preds['IND'])), 8, replace=False).astype(int)
        inds = np.sort(inds)
    samples = np.array(args[0].preds['IND']).astype(int)[inds]

    subfigs = fig.subfigures(nrows=len(args) + 1, ncols=1)

    subfigs[0].suptitle("Ground Truth", fontweight="bold")
    axes = subfigs[0].subplots(nrows=1, ncols=8)
    for j in range(len(axes)):
        _ind = np.where(args[0].gt_ind == samples[j])
        img = axes[j].imshow(np.squeeze(args[0].gt[_ind]), vmin=0, vmax=1)
        axes[j].axis('off')
        axes[j].set_title(f"#{samples[j]}")

    for i, ar in enumerate(args):
        subfigs[i+1].suptitle(ar.name, fontweight="bold")
        axes = subfigs[i+1].subplots(nrows=1, ncols=8)
        for j in range(len(axes)):
            _ind = np.where(ar.preds['IND'] == samples[j])
            img = axes[j].imshow(np.squeeze(ar.resized[_ind]), vmin=0, vmax=1)
            axes[j].axis('off')
            axes[j].set_title(f"#{samples[j]}")
    plt.show()
    filename = f"comparison_visual.jpg"
    return fig, filename


