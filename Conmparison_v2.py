import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2
from misc import plot_settings


class ResultCalculator:
    zero = False

    def __init__(self, name, pred_path, gt=None):
        self.name = name
        print(f"{self.name} loading...")
        self.preds: dict = np.load(pred_path, allow_pickle=True).item() if pred_path else None
        self.tags = self.preds['TAG'] if pred_path else None
        if self.preds:
            print("{name} loaded Estimates of {pred_img.shape} as {pred_img.dtype}".format(
                name=self.name,
                pred_img=np.array(self.preds['S_PRED'] if 'S_PRED' in self.preds.keys() else self.preds['PRED']))
            )

        self.gt = gt
        self.image_size = (128, 226)  # in rows * columns
        self.resized = np.zeros((len(self.tags), *self.image_size)) if pred_path else None
        self.result = np.zeros_like(self.tags) if pred_path else None
        self.loss = F.mse_loss

    def resize(self):
        print(f"{self.name} resizing...", end='')
        for i in range(len(self.tags)):
            self.resized[i] = cv2.resize(
                np.squeeze(self.preds['S_PRED'][i] if 'S_PRED' in self.preds.keys() else self.preds['PRED'][i]),
                (self.image_size[1], self.image_size[0]))
        print("Done!")

    def calculate_loss(self):
        print(f"{self.name} calculating loss...", end='')
        for i, tag in enumerate(self.tags):
            # Locate ground truth by ind and take
            take, ind = tag[0], tag[-1]
            _ind = np.where(self.gt['tag'][:, -1] == ind)
            _take = np.where(self.gt['tag'][_ind][:, 0] == take)

            pred = torch.from_numpy(self.resized[i])
            self.result[i] = F.mse_loss(pred, torch.from_numpy(self.gt['tag'][_ind][_take]))
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