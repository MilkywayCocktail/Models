import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2
from misc import plot_settings
from PIL import Image
from scipy import signal


class ResultCalculator:
    zero = False

    def __init__(self, name, pred_path, gt=None):
        self.name = name
        print(f"{self.name} loading...")
        self.preds: dict = np.load(pred_path, allow_pickle=True).item() if pred_path else None
        self.tags = np.array(self.preds['TAG']) if pred_path else None
        if self.preds:
            print("{name} loaded Estimates of {pred_img.shape} as {pred_img.dtype}".format(
                name=self.name,
                pred_img=np.array(self.preds['S_PRED'] if 'S_PRED' in self.preds.keys() else self.preds['PRED']))
            )

        self.gt = gt['rimg']
        self.gt_tag = gt['tag']
        self.image_size = (128, 226)  # in rows * columns
        
        self.resized = np.zeros((len(self.tags), *self.image_size), dtype=float) if pred_path else None
        self.matched = np.zeros((len(self.tags), *self.image_size), dtype=float) if pred_path else None
        
        self.result = np.zeros_like(self.tags, dtype=float) if pred_path else None
        self.result_matched = np.zeros_like(self.tags, dtype=float) if pred_path else None
        self.deviation = np.zeros((len(self.tags), 3), dtype=float) if pred_path else None
        
        self.loss = F.mse_loss

    def resize(self):
        print(f"{self.name} resizing...", end='')
        for i in range(len(self.tags)):
            self.resized[i] = cv2.resize(
                np.squeeze(self.preds['S_PRED'][i] if 'S_PRED' in self.preds.keys() else self.preds['PRED'][i]),
                (self.image_size[1], self.image_size[0]))
        print("Done!")
        
    def find_real_ind(self, iind):
        take, ind = self.tags[iind][0], self.tags[iind][-1]
        _ind = np.where(self.gt_tag[:, -1] == ind)
        _take = np.where(self.gt_tag[_ind][:, 0] == take)
        return _take, _ind

    def calculate_loss(self):
        print(f"{self.name} calculating loss...", end='')
        for i, tag in enumerate(self.tags):
            # Find gt tag from pred tag
            take, ind = tag[0], tag[-1]
            _ind = np.where(self.gt_tag[:, -1] == ind)
            _take = np.where(self.gt_tag[_ind][:, 0] == take)

            pred = torch.from_numpy(self.resized[i])
            self.result[i] = F.mse_loss(pred, torch.from_numpy(self.gt[_ind][_take]))
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
    
    def matching_mae(self, scale=0.3):
        print(f"{self.name} calculating 2D correlation...", end='')
        for i, (im, tag) in enumerate(zip(self.resized, self.tags)):
            
            _im = Image.fromarray((im * 255).astype(np.uint8))
            im = _im.resize((int(self.image_size[1]*scale), int(self.image_size[0]*scale)),Image.BILINEAR)
            im = np.array(im).astype(float)
            
            take, ind = self.find_real_ind(i)
            gt = np.squeeze(self.gt[ind][take])
            
            _gt = Image.fromarray((gt * 255).astype(np.uint8))
            gt = _gt.resize((int(self.image_size[1]*scale), int(self.image_size[0]*scale)),Image.BILINEAR)
            gt = np.array(gt).astype(float)
        
            tmp = signal.correlate2d(gt, im, mode="full")
            y, x = np.unravel_index(np.argmax(tmp), tmp.shape)
            y = int(y / scale) - self.image_size[0]
            x = int(x / scale) - self.image_size[1]
        
            im_re = _im.rotate(0, translate = (x, y))
            im_re = np.array(im_re).astype(float) / 255.
            gt_re = _gt.resize((int(self.image_size[1]), int(self.image_size[0])),Image.BILINEAR)
            gt_re = np.array(gt_re).astype(float) / 255.
            
            self.result_matched[i] = np.mean(np.abs(gt_re - im_re))
            self.matched[i] = im_re
            self.deviation[i] = np.array([x, y, np.sqrt(x**2+y**2)])

        print("Done!")


class BBXResultCalculator(ResultCalculator):
    def __init__(self, *args, **kwargs):
        super(BBXResultCalculator, self).__init__(*args, **kwargs)

        self.bbx = np.array(self.preds['S_BBX'])
        self.center = np.zeros((len(self.preds['S_BBX']), 2))
        self.depth = np.array(self.preds['S_DPT'])

        self.min_area = 0
        self.fail_count = 0
        self.fail_ind = []

    def resize(self):
        print("Reconstructing...", end='')
        for i in range(len(self.bbx)):
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
                    self.center[i] = np.array([y0 + int(h0 / 2), x0 + int(w0 / 2)])

                    # In case of reversing (x1, y1) and (x2, y2)
                    if w0 < 0 and h0 < 0:
                        x0 = int(x2 * 226)
                        y0 = int(y2 * 128)
                        w0 = int((x1 - x2) * 226)
                        h0 = int((y1 - y2) * 128)
                        self.center[i] = np.array([y0 + int(h0 / 2), x0 + int(w0 / 2)])

                    try:
                        subject1 = cv2.resize(subject, (w0, h0))
                        for x_ in range(w0):
                            for y_ in range(h0):
                                self.resized[i, y0 + y_, x0 + x_] = subject1[y_, x_]

                    except Exception as e:
                        print(e)
                        print(x0, y0, w0, h0)
                        self.fail_count += 1
                        self.fail_ind.append(i)
        print("Done")
        print(f"Reconstruction finished. Failure count = {self.fail_count}")

    def plot_example(self, inds=None, title=None):
        fig = plot_settings()
        fig.suptitle(f"{self.name} Reconstruction Examples" if not title else title)

        subfigs = fig.subfigures(nrows=4, ncols=1)

        plot_terms = {'Cropped Ground Truth': self.preds['GT'],
                      'Cropped Estimates': self.preds['S_PRED'],
                      'Raw Ground Truth': self.gt,
                      'Raw Estimates': self.resized}

        if not inds:
            inds = np.random.choice(np.arange(len(self.preds['TAG']), dtype=int), 8, replace=False)

        for i, (key, value) in enumerate(plot_terms.items()):
            subfigs[i].suptitle(key, fontweight="bold")
            axes = subfigs[i].subplots(nrows=1, ncols=len(inds))
            for j, iind in enumerate(inds):
                take, ind = self.find_real_ind(iind)
                img = axes[j].imshow(np.squeeze(value[ind][take]) if key == 'Raw Ground Truth'
                                     else np.squeeze(value[iind]), vmin=0, vmax=1)
                if key == 'Raw Ground Truth':
                    x1, y1, x2, y2 = self.preds['GT_BBX'][iind]
                    x = int(x1 * 226)
                    y = int(y1 * 128)
                    w = int((x2 - x1) * 226)
                    h = int((y2 - y1) * 128)
                    axes[j].add_patch(Rectangle((x, y), w, h, edgecolor='pink', fill=False, lw=3))
                elif key == 'Raw Estimates':
                    x1, y1, x2, y2 = self.preds['S_BBX'][iind]
                    x = int(x1 * 226)
                    y = int(y1 * 128)
                    w = int((x2 - x1) * 226)
                    h = int((y2 - y1) * 128)
                    axes[j].add_patch(Rectangle((x, y), w, h, edgecolor='orange', fill=False, lw=3))
                axes[j].axis('off')
                axes[j].set_title(f"{'-'.join(map(str, map(int, self.tags[iind])))}")
        plt.show()
        filename = f"{self.name}_Reconstruct.jpg"
        return fig, filename


class CenterResultCalculator(ResultCalculator):
    def __init__(self, *args, **kwargs):
        super(CenterResultCalculator, self).__init__(*args, **kwargs)

        self.center = np.array(self.preds['S_CTR'])
        self.depth = np.array(self.preds['S_DPT'])

        self.fail_count = 0
        self.fail_ind = []

    def resize(self):
        print("Reconstructing...", end='')

        for i, (x, y) in enumerate(self.center):
            x = int(x * 226)
            y = int(y * 128)
            # Image slightly larger image than raw size
            img = np.squeeze(self.preds['S_PRED'][i]) * np.squeeze(self.depth[i])
            new_img = np.zeros((140, 248), dtype=float)
            new_img[6:-6, 60:-60] = img

            try:
                # Roll to estimated position
                new_img = np.roll(new_img, y-64, axis=0)
                new_img = np.roll(new_img, x-113, axis=1)
                # Crop out raw image
                self.resized[i] = new_img[6:-6, 11:-11]

            except Exception as e:
                print(e)
                print(x, y)
                self.fail_count += 1
                self.fail_ind.append(i)

        print("Done")
        print(f"Reconstruction finished. Failure count = {self.fail_count}")

    def plot_example(self, inds=None, title=None):
        fig = plot_settings()
        fig.suptitle(f"{self.name} Reconstruction Examples" if not title else title)

        subfigs = fig.subfigures(nrows=4, ncols=1)

        plot_terms = {
            'Cropped Ground Truth': self.preds['GT'],
            'Cropped Estimates'   : self.preds['S_PRED'],
            'Raw Ground Truth'    : self.gt,
            'Raw Estimates'       : self.resized}

        if not inds:
            inds = np.random.choice(np.arange(len(self.preds['TAG']), dtype=int), 8, replace=False)

        for i, (key, value) in enumerate(plot_terms.items()):
            subfigs[i].suptitle(key, fontweight="bold")
            axes = subfigs[i].subplots(nrows=1, ncols=len(inds))
            for j, iind in enumerate(inds):
                take, ind = self.find_real_ind(iind)
                img = axes[j].imshow(np.squeeze(value[ind][take]) if key == 'Raw Ground Truth'
                                    else np.squeeze(value[iind]), vmin=0, vmax=1)

                if key == 'Raw Ground Truth':
                    x1, y1 = self.preds['GT_CTR'][iind]
                    x1 = int(x1 * 226)
                    y1 = int(y1 * 128)
                    axes[j].scatter(x1, y1, c='red', marker=(5, 1), alpha=0.5, linewidths=5, label='GT_CTR')
                elif key == 'Raw Estimates':
                    x2, y2 = self.preds['S_CTR'][iind]
                    x2 = int(x2 * 226)
                    y2 = int(y2 * 128)
                    axes[j].scatter(x2, y2, c='red', marker=(5, 1), alpha=0.5, linewidths=5, label='S_CTR')
                axes[j].axis('off')
                axes[j].set_title(f"{'-'.join(map(str, map(int, self.tags[iind])))}")
        plt.show()
        filename = f"{self.name}_Reconstruct.jpg"
        return fig, filename


class ZeroEstimates(ResultCalculator):
    zero = True

    def __init__(self, *args, **kwargs):
        super(ZeroEstimates, self).__init__(*args, **kwargs)

        print(f"{self.name} loaded Zero Estimates")

        self.resized = np.zeros((len(self.gt), *self.image_size))
        self.result = np.zeros(len(self.gt), dtype=float)

    def resize(self):
        print(f"{self.name} resized")


def gather_plot(*args: ResultCalculator, title=None):
    fig = plot_settings()
    fig.suptitle('Comparison Results' if not title else title)

    bins = np.linspace(np.min([np.min(ar.result) for ar in args]), np.max([np.max(ar.result) for ar in args]), 50)

    for i, ar in enumerate(args):

        hist_, bin_edges = np.histogram(ar.result, bins)
        width = (bin_edges[1] - bin_edges[0]) * 0.8
        cdf = np.cumsum(hist_ / sum(hist_))
        if not ar.zero:
            plt.bar(bin_edges[1:], hist_ / max(hist_), width=width, label=ar.name, zorder=i)
            plt.plot(bin_edges[1:], cdf, '-*', label=ar.name, zorder=i + len(args))
        else:
            plt.bar(bin_edges[1:], hist_ / max(hist_), width=width * 0.4, label=ar.name, hatch='-', zorder=len(args))
            plt.plot(bin_edges[1:], cdf, '-.', marker='o', label=ar.name, zorder=2 * len(args))

    ax = plt.gca()
    ax.fill_between(np.arange(0, np.max([np.max(ar.result) for ar in args]), 0.01), 1.02,
                    color='white', alpha=0.5, zorder=len(args) + 0.5)
    plt.title('Test PDF-CDF', fontweight="bold")
    plt.xlabel('Per-sample Loss')
    plt.ylabel('Frequency')
    plt.grid()
    plt.legend()
    plt.show()
    filename = f"comparison_CDF.jpg"

    return fig, filename


def visualization(*args: ResultCalculator, univ_gt=None, inds=None, figsize=(20, 10), title=None, matched=False):
    fig = plot_settings(figsize)
    mch = ' - Matched' if matched else None
    fig.suptitle(f'Comparison Visualization{mch}' if not title else title)

    assert univ_gt
    if inds is None:
        inds = univ_gt['tag'][np.random.choice(np.arange(len(univ_gt['tag']), dtype=int), 8, replace=False)]

    subfigs = fig.subfigures(nrows=len(args) + 1, ncols=1)

    subfigs[0].suptitle("Ground Truth", fontweight="bold")
    axes = subfigs[0].subplots(nrows=1, ncols=len(inds))
    for j, iind in enumerate(inds):
        take, ind = iind[0], iind[-1]
        _ind = np.where(univ_gt['tag'][:, -1] == ind)
        _take = np.where(univ_gt['tag'][_ind][:, 0] == take)
        img = axes[j].imshow(np.squeeze(univ_gt['rimg'][_ind][_take]), vmin=0, vmax=1)
        axes[j].axis('off')
        axes[j].set_title(f"{'-'.join(map(str, map(int, iind)))}")

    for i, ar in enumerate(args):
        if not ar.zero:
            subfigs[i + 1].suptitle(ar.name, fontweight="bold")
            axes = subfigs[i + 1].subplots(nrows=1, ncols=len(inds))
            for j, iind in enumerate(inds):
                take, ind = iind[0], iind[-1]
                _ind = np.where(ar.tags[:, -1] == ind)
                _take = np.where(ar.tags[_ind][:, 0] == take)
                img = axes[j].imshow(np.squeeze(ar.matched[_ind][_take] if matched else ar.resized[_ind][_take]), vmin=0, vmax=1)
                axes[j].axis('off')
                axes[j].set_title(f"{'-'.join(map(str, map(int, iind)))}")
                if matched:
                    st = [ar.center[_ind][_take][..., 1], ar.center[_ind][_take][..., 0]]
                    ed = [st[0] + ar.deviation[_ind][_take][..., 0], st[1] + ar.deviation[_ind][_take][..., 1]]
                    axes[j].plot([st[0], ed[0]], [st[1], ed[1]], color='red', linewidth=2)
    plt.show()
    filename = f"comparison_visual.jpg"
    return fig, filename, inds
