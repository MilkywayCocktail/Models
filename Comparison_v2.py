import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2
from misc import plot_settings
from PIL import Image
from scipy import signal
import os


class ResultCalculator:
    zero = False

    def __init__(self, name, pred_path, gt=None):
        self.name = name
        print(f"{self.name} loading...")
        self.preds: dict = np.load(pred_path, allow_pickle=True).item() if pred_path else None
        self.tags = np.array(self.preds['TAG']).astype(int) if pred_path else None
        if self.preds:
            print("{name} loaded Estimates of {pred_img.shape} as {pred_img.dtype}".format(
                name=self.name,
                pred_img=np.array(self.preds['S_PRED'] if 'S_PRED' in self.preds.keys() else self.preds['PRED']))
            )

        self.gt = gt['rimg']
        self.gt_tag = gt['tag'].astype(int)
        self.image_size = (128, 226)  # in rows * columns
        
        self.resized = np.zeros((len(self.tags), *self.image_size), dtype=float) if pred_path else None
        self.matched = np.zeros((len(self.tags), *self.image_size), dtype=float) if pred_path else None
        self.center = np.zeros((len(self.tags), 2), dtype=int) if pred_path else None
        
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
        
    @staticmethod
    def find_real_ind(iind: int, source, target):
        """Find the real index of a sample in target from the tag of source.\n

        Args:
            iind (int): index of sample
            source (ndarray): source tags
            target (ndarray): target tags

        Returns:
            _type_: _description_
        """
        take, ind = source[iind][0], source[iind][-1]
        _ind = np.where(target[:, -1] == ind)
        _take = np.where(target[_ind][:, 0] == take)
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

    def calculate_cdf(self, show_fig=True):
        print(f"{self.name} calculating histograms...", end='')
        hist, bin_edges = np.histogram(self.result)
        print("Done!")
        fig = plot_settings()
        fig.suptitle(f"{self.name} Loss on Raw Images")
        filename = f"{self.name}_CDF.jpg"
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
        if show_fig:
            plt.show()
            
        return {filename: fig}
    
    def matching_mae(self, scale=0.3):
        print(f"{self.name} calculating 2D correlation...", end='')
        for i, im in enumerate(self.resized):
            # Find center
            (T, timg) = cv2.threshold((np.squeeze(im) * 255).astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(timg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) != 0:
                contour = max(contours, key=lambda x: cv2.contourArea(x))
                area = cv2.contourArea(contour)

                if area < 0:
                    pass

                else:
                    x, y, w, h = cv2.boundingRect(contour)
                    self.center[i][0] = int(x + w / 2)
                    self.center[i][1] = int(y + h / 2)
            
            _im = Image.fromarray((im * 255).astype(np.uint8))
            im = _im.resize((int(self.image_size[1]*scale), int(self.image_size[0]*scale)),Image.BILINEAR)
            im = np.array(im).astype(float)
            
            take, ind = self.find_real_ind(i, self.tags, self.gt_tag)
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
        
    def plot_example(self, inds=None, title=None, matched=False):
        fig = plot_settings()
        fig.suptitle(f"{self.name} Reconstruction Examples" if not title else title)
        filename = f"{self.name}_Reconstruct.jpg"

        plot_terms = {'Raw Ground Truth': self.gt,
                      'Raw Estimates': self.resized}
        if matched:
            plot_terms['Matched Estimates'] = self.matched
            filename = f"{self.name}_Reconstruct_Matched.jpg"
        subfigs = fig.subfigures(nrows=len(list(plot_terms.keys())), ncols=1)

        if not inds:
            inds = np.random.choice(np.arange(len(self.preds['TAG']), dtype=int), 8, replace=False)

        for i, (key, value) in enumerate(plot_terms.items()):
            subfigs[i].suptitle(key, fontweight="bold")
            axes = subfigs[i].subplots(nrows=1, ncols=len(inds))
            for j, iind in enumerate(inds):
                take, ind = self.find_real_ind(iind, self.tags, self.gt_tag)
                axes[j].imshow(np.squeeze(self.gt[ind][take]) if key == 'Raw Ground Truth'
                               else np.squeeze(value[iind]), vmin=0, vmax=1)
                if key == 'Matched Estimates':
                    st = [self.center[iind][..., 1], self.center[iind][..., 0]]
                    ed = [st[0] + self.deviation[iind][..., 0], st[1] + self.deviation[iind][..., 1]]
                    axes[j].plot([st[0], ed[0]], [st[1], ed[1]], color='red', linewidth=2)
                    
                axes[j].axis('off')
                axes[j].set_title(f"{'-'.join(map(str, map(int, self.tags[iind])))}")
        plt.show()
        
        return {filename: fig}


class BBXResultCalculator(ResultCalculator):
    def __init__(self, *args, **kwargs):
        super(BBXResultCalculator, self).__init__(*args, **kwargs)

        self.bbx = np.array(self.preds['S_BBX'])
        self.depth = np.array(self.preds['S_DPT'])

        self.min_area = 0
        self.fail_count = 0
        self.fail_ind = []

    def resize(self):
        print(f"{self.name} reconstructing...", end='')
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
                    
                    x0 = int(min(x1, x2) * 226)
                    y0 = int(min(y1, y2) * 128)
                    w0 = int(abs((x2 - x1) * 226))
                    h0 = int(abs((y2 - y1) * 128))

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
        print(f" Reconstruction finished. Failure count = {self.fail_count}")

    def plot_example(self, inds=None, title=None, matched=False):
        fig = plot_settings()
        fig.suptitle(f"{self.name} Reconstruction Examples" if not title else title)
        filename = f"{self.name}_Reconstruct.jpg"
       
        plot_terms = {'Cropped Ground Truth': self.preds['GT'],
                      'Cropped Estimates': self.preds['S_PRED'],
                      'Raw Ground Truth': self.gt,
                      'Raw Estimates': self.resized}
        if matched:
            plot_terms['Matched Estimates'] = self.matched
            filename = f"{self.name}_Reconstruct_Matched.jpg"
        subfigs = fig.subfigures(nrows=len(list(plot_terms.keys())), ncols=1)

        if not inds:
            inds = np.random.choice(np.arange(len(self.preds['TAG']), dtype=int), 8, replace=False)

        for i, (key, value) in enumerate(plot_terms.items()):
            subfigs[i].suptitle(key, fontweight="bold")
            axes = subfigs[i].subplots(nrows=1, ncols=len(inds))
            for j, iind in enumerate(inds):
                take, ind = self.find_real_ind(iind, self.tags, self.gt_tag)
                axes[j].imshow(np.squeeze(self.gt[ind][take]) if key == 'Raw Ground Truth'
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
                elif key == 'Matched Estimates':
                    st = [self.center[iind][0], self.center[iind][1]]
                    ed = [st[0] + self.deviation[iind][..., 0], st[1] + self.deviation[iind][..., 1]]
                    axes[j].plot([st[0], ed[0]], [st[1], ed[1]], color='red', linewidth=2)
                    
                axes[j].axis('off')
                axes[j].set_title(f"{'-'.join(map(str, map(int, self.tags[iind])))}")
        plt.show()
        
        return {filename: fig}


class CenterResultCalculator(ResultCalculator):
    def __init__(self, *args, **kwargs):
        super(CenterResultCalculator, self).__init__(*args, **kwargs)

        self.depth = np.array(self.preds['S_DPT'])

        self.fail_count = 0
        self.fail_ind = []

    def resize(self):
        print(f"{self.name} reconstructing...", end='')

        for i, (x, y) in enumerate(self.preds['S_CTR']):
            x = int(x * 226)
            y = int(y * 128)
            # Pad cropped image into raw-image size
            img = np.squeeze(self.preds['S_PRED'][i]) * self.depth[i]
            new_img = np.pad(img, [(0, 0), (49, 49)], 'constant', constant_values=0)

            try:
                # Roll to estimated position
                new_img = np.roll(new_img, y-64, axis=0)
                new_img = np.roll(new_img, x-113, axis=1)
                self.resized[i] = new_img

            except Exception as e:
                print(e)
                print(x, y)
                self.fail_count += 1
                self.fail_ind.append(i)

        print("Done")
        print(f" Reconstruction finished. Failure count = {self.fail_count}")

    def plot_example(self, inds=None, title=None, matched=False):
        fig = plot_settings()
        fig.suptitle(f"{self.name} Reconstruction Examples" if not title else title)
        filename = f"{self.name}_Reconstruct.jpg"

        plot_terms = {
            'Cropped Ground Truth': self.preds['GT'],
            'Cropped Estimates'   : self.preds['S_PRED'],
            'Raw Ground Truth'    : self.gt,
            'Raw Estimates'       : self.resized}
        
        if matched:
            plot_terms['Matched Estimates'] = self.matched
            filename = f"{self.name}_Reconstruct_Matched.jpg"
        subfigs = fig.subfigures(nrows=len(list(plot_terms.keys())), ncols=1)
        
        if not inds:
            inds = np.random.choice(np.arange(len(self.preds['TAG']), dtype=int), 8, replace=False)

        for i, (key, value) in enumerate(plot_terms.items()):
            subfigs[i].suptitle(key, fontweight="bold")
            axes = subfigs[i].subplots(nrows=1, ncols=len(inds))
            for j, iind in enumerate(inds):
                take, ind = self.find_real_ind(iind, self.tags, self.gt_tag)
                axes[j].imshow(np.squeeze(value[ind][take]) if key == 'Raw Ground Truth'
                                    else np.squeeze(value[iind]), vmin=0, vmax=1)

                if key == 'Raw Ground Truth':
                    x, y= self.preds['GT_CTR'][iind]
                    x = int(x * 226)
                    y = int(y * 128)
                    axes[j].scatter(x, y, c='red', marker=(5, 1), alpha=0.5, linewidths=5, label='GT_CTR')
                elif key == 'Raw Estimates':
                    x, y= self.preds['S_CTR'][iind]
                    x = int(x * 226)
                    y = int(y * 128)
                    axes[j].scatter(x, y, c='red', marker=(5, 1), alpha=0.5, linewidths=5, label='S_CTR')
                elif key == 'Matched Estimates':
                    st = [self.center[iind][0], self.center[iind][1]]
                    ed = [st[0] + self.deviation[iind][..., 0], st[1] + self.deviation[iind][..., 1]]
                    axes[j].plot([st[0], ed[0]], [st[1], ed[1]], color='red', linewidth=2)
                
                axes[j].axis('off')
                axes[j].set_title(f"{'-'.join(map(str, map(int, self.tags[iind])))}")
        plt.show()

        return {filename: fig}


class ZeroEstimates(ResultCalculator):
    zero = True

    def __init__(self, *args, **kwargs):
        super(ZeroEstimates, self).__init__(*args, **kwargs)

        print(f"{self.name} loaded Zero Estimates")

        self.resized = np.zeros((len(self.gt), *self.image_size))
        self.matched = np.zeros((len(self.gt), *self.image_size))
        self.center = np.zeros((len(self.gt), 2))
        self.result = np.zeros(len(self.gt), dtype=float)
        self.result_matched = np.zeros(len(self.gt), dtype=float)
        self.deviation = np.zeros((len(self.gt), 3))
        self.tags = self.gt_tag

    def resize(self):
        print(f"{self.name} resized")


class GatherPlotCDF:
    def __init__(self, subjects:dict):
        self.subjects = subjects
        self.count = 0
        
    def __call__(self, scope='all', item='mse', title=None):
        scope = list(self.subjects.keys()) if scope=='all' else scope
        fig = plot_settings((20, 10))
        for i, sub in enumerate(scope):
            if item == 'mse':
                mmin = [ar.result.min() for ar in self.subjects.values()]
                mmax = [ar.result.max() for ar in self.subjects.values()]
                bins = np.linspace(np.min(mmin), np.max(mmax), 50)
                hist_, bin_edges = np.histogram(self.subjects[sub].result, bins)
                filename = f"Comparison_CDF_MSE_{self.count}.jpg"
            elif item == 'matched_mae':
                mmin = [ar.result_matched.min() for ar in self.subjects.values()]
                mmax = [ar.result_matched.max() for ar in self.subjects.values()]
                bins = np.linspace(np.min(mmin), np.max(mmax), 50)
                hist_, bin_edges = np.histogram(self.subjects[sub].result_matched, bins)
                filename = f"Comparison_CDF_MatchedMAE_{self.count}.jpg"
            elif item == 'deviation':
                mmin = [ar.deviation[..., -1].min() for ar in self.subjects.values()]
                mmax = [ar.deviation[..., -1].max() for ar in self.subjects.values()]
                bins = np.linspace(np.min(mmin), np.max(mmax), 50)
                hist_, bin_edges = np.histogram(self.subjects[sub].deviation[..., -1], bins)
                filename = f"Comparison_CDF_Deviation_{self.count}.jpg"
            width = (bin_edges[1] - bin_edges[0]) * 0.8
            cdf = np.cumsum(hist_ / sum(hist_))
            if not self.subjects[sub].zero:
                plt.bar(bin_edges[1:], hist_ / max(hist_), width=width, label=sub, zorder=i)
                plt.plot(bin_edges[1:], cdf, '-*', label=sub, zorder=i + len(scope))
            else:
                plt.bar(bin_edges[1:], hist_ / max(hist_), width=width * 0.4, label=sub, hatch='-', zorder=len(scope))
                plt.plot(bin_edges[1:], cdf, '-.', marker='o', label=sub, zorder=2 * len(scope))

        ax = plt.gca()
        ax.fill_between(np.arange(0, np.max(mmax), 0.01), 1.02,
                        color='white', alpha=0.5, zorder=len(scope) + 0.5)
        plt.title('Test PDF-CDF', fontweight="bold")
        plt.xlabel(f'Per-sample {item}')
        plt.ylabel('Frequency')
        plt.grid()
        plt.legend()
        plt.show()
        self.count += 1
        return {filename: fig}

class ResultVisualize:
    def __init__(self, subjects:dict, univ_gt):
        self.subjects = subjects
        self.univ_gt = univ_gt
        self.count = 0
        self.inds = None
        self.figsize = (20, 20)
        self.find_real_ind = ResultCalculator.find_real_ind
        
    def __call__(self, scope='all', inds=None, matched=False, title=None):
        scope = list(self.subjects.keys()) if scope=='all' else scope
        
        fig = plot_settings(self.figsize)
        mch = ' - Matched' if matched else ''
        fig.suptitle(f'Comparison Visualization{mch}' if not title else title)
        filename = f"Comparison_Visual_Matched_{self.count}.jpg" if matched else f"Comparison_Visual_{self.count}.jpg"
        assert self.univ_gt
        if inds is None:
            if self.inds is None:
                inds = np.random.choice(np.arange(len(self.univ_gt['tag']), dtype=int), 8, replace=False)
                self.inds = inds
            else:
                inds = self.inds

        subfigs = fig.subfigures(nrows=len(scope) + 1, ncols=1)
        subfigs[0].suptitle("Ground Truth", fontweight="bold")
        axes = subfigs[0].subplots(nrows=1, ncols=len(inds))
        for j, iind in enumerate(inds):
            axes[j].imshow(np.squeeze(self.univ_gt['rimg'][iind]), vmin=0, vmax=1)
            axes[j].axis('off')
            axes[j].set_title(f"{'-'.join(map(str, map(int, self.univ_gt['tag'][iind])))}")

        for i, sub in enumerate(scope):
            if not self.subjects[sub].zero:
                subfigs[i + 1].suptitle(self.subjects[sub].name, fontweight="bold")
                axes = subfigs[i + 1].subplots(nrows=1, ncols=len(inds))
                for j, iind in enumerate(inds):
                    take, ind = self.find_real_ind(iind, self.univ_gt['tag'], self.subjects[sub].tags)
                    axes[j].imshow(np.squeeze(self.subjects[sub].matched[ind][take] if matched else self.subjects[sub].resized[ind][take]), vmin=0, vmax=1)
                    axes[j].axis('off')
                    axes[j].set_title(f"{'-'.join(map(str, map(int, self.univ_gt['tag'][iind])))}")
                    if matched:
                        st = [self.subjects[sub].center[ind][take][..., 0], 
                              self.subjects[sub].center[ind][take][..., 1]]
                        ed = [st[0] + self.subjects[sub].deviation[ind][take][..., 0], 
                              st[1] + self.subjects[sub].deviation[ind][take][..., 1]]
                        axes[j].plot([st[0], ed[0]], [st[1], ed[1]], color='red', linewidth=2)
        plt.show()
        self.count += 1
        return {filename: fig}
    
class ResultProcess:
    def __init__(self, subjects:dict, most_gt, least_gt):
        self.subjects = subjects
        self.most_gt = most_gt
        self.least_gt = least_gt
        self._gatherplot = GatherPlotCDF(subjects=self.subjects)
        self._visualization = ResultVisualize(subjects=self.subjects, univ_gt=self.least_gt)
        self.figs: dict = {}
        
    def load_preds(self):
        for sub, path in self.subjects.items():
            if 'Center' in sub:
                self.subjects[sub] = CenterResultCalculator(sub, path, self.most_gt)
            elif 'BBX' in sub:
                self.subjects[sub] = BBXResultCalculator(sub, path, self.most_gt)
            elif 'Zero' in sub:
                self.subjects[sub] = ZeroEstimates(sub, None,self.most_gt)
            else:
                self.subjects[sub] = ResultCalculator(sub, path, self.most_gt)
                
    def resize(self, scope='all', show_fig=False, **kwargs):
        scope = list(self.subjects.keys()) if scope=='all' else scope
        
        for sub in scope:
            self.subjects[sub].resize()
            if show_fig:
                self.figs.update(self.subjects[sub].plot_example(**kwargs))
                
    def matching_mae(self, scope='all', scale=0.3, show_fig=False, **kwargs):
        scope = list(self.subjects.keys()) if scope=='all' else scope
        
        for sub in scope:
            self.subjects[sub].matching_mae(scale=scale)
            if show_fig:
                self.figs.update(self.subjects[sub].plot_example(matched=True, **kwargs))
                
    def calculate_loss(self, scope='all'):
        scope = list(self.subjects.keys()) if scope=='all' else scope
        
        for sub in scope:
            self.subjects[sub].calculate_loss()
    
    def gatherplot(self, *args, **kwargs):
        self.figs.update(self._gatherplot(*args, **kwargs))
        
    def visualize(self, *args, **kwargs):
        self.figs.update(self._visualization(*args, **kwargs))
        
    def save_figs(self, notion):
        save_path = f'../saved/{notion}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for filename, fig in self.figs.items():
            fig.savefig(f"{save_path}{filename}")