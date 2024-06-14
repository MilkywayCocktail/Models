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
import pandas as pd
from tqdm.notebook import tqdm

class Estimates:
    def __init__(self, name, path=None, modality={'GT', 'PRED', 'S_PRED', 'GT_BBX', 'S_BBX', 'GT_CTR', 'S_CTR', 'GT_DPT', 'S_DPT', 'TAG'}):
        self.name = name
        self.path = path
        self.preds: dict = {}
        self.len = 0
        if path:
            self.load_preds(modality)
            self.tags, self.seg_tags = self.structurize_tags(self.tags)
        
    def __len__(self):
        return self.len
                            
    def load_preds(self, modality):
        paths = os.walk(self.path)
        for path, _, file_lst in paths:
            for file_name in file_lst:
                file_name_, ext = os.path.splitext(file_name)
                if ext == '.npy':
                    keypos = file_name_.find('preds')
                    if keypos != -1:
                        key = file_name_[keypos+6:]
                        if modality and key in modality or not modality:
                            self.preds[key] = np.load(os.path.join(path, file_name), mmap_mode='r') 
                            self.len = self.preds[key].shape[0]
                            print(f"{self.name} loaded {key} of {self.preds[key].shape} as {self.preds[key].dtype}")
        self.tags = self.preds['TAG']
    
    @staticmethod
    def structurize_tags(tags):
        tags = pd.DataFrame(tags, columns=['take', 'group', 'segment', 'sample'])
        tags = tags.sort_values(by=['take', 'group', 'segment', 'sample'])
        
        seg_tags = pd.DataFrame(columns=['take', 'group', 'segment', 'samples', 'inds'], dtype=object)
        takes = set(tags.loc[:, 'take'].values)

        for take in takes:
            groups = set(tags[tags['take']==take].group.values)
            for group in groups:
                segments = set(tags[(tags['take']==take) & (tags['group']==group)].segment.values)
                for segment in segments:
                    selected = tags[(tags['take']==take) & (tags['group']==group) & (tags['segment']==segment)]
                    selected = selected.sort_values(by=['sample'])
                    new_rec = [take, group, segment, selected['sample'].tolist(), selected.index.tolist()]
                    seg_tags.loc[len(seg_tags)] = new_rec
        return tags, seg_tags

class ResultCalculator(Estimates):
    zero = False
    
    def __init__(self, gt=None, *args, **kwargs):
        super(ResultCalculator, self).__init__(*args, **kwargs)

        self.gt = gt['rimg']
        self.gt_tags, self.gt_seg_tags = self.structurize_tags(gt['tag'])
        if self.preds:
            self.img_pred = self.preds['S_PRED'] if 'S_PRED' in self.preds.keys() else self.preds['PRED']
        self.image_size = (128, 226)  # in rows * columns
        self.postprocessed = False
        
        _len = len(self.gt_tags) if self.zero else self.len
        _seglen = len(self.gt_seg_tags) if self.zero else len(self.seg_tags)
        
        self.resized = {'vanilla': np.zeros((_len, *self.image_size), dtype=float),
                        'postprocessed': np.zeros((_len, *self.image_size), dtype=float)}
        self.matched = {'vanilla': np.zeros((_len, *self.image_size), dtype=float),
                        'postprocessed': np.zeros((_len, *self.image_size), dtype=float)}
        self.center = {'vanilla': np.zeros((_len, 2), dtype=int),
                       'postprocessed': np.zeros((_len, 2), dtype=int)}
        
        res = pd.DataFrame(np.zeros((_len, 5), dtype=float),
                           columns=['mse', 'matched_mae', 'dev_x', 'dev_y', 'deviation'])
        segres = pd.DataFrame(np.zeros((_seglen, 5), dtype=float),
                              columns=['mse', 'matched_mae', 
                                       'dev_x', 'dev_y', 'deviation'])
        segments = pd.DataFrame(np.zeros((_seglen, 3)), dtype=int, columns=['take', 'group', 'segment'])
        
        self.result = pd.concat({'vanilla': res, 'postprocessed': res}, axis=1)
        self.seg_result = pd.concat({'segments': segments, 'vanilla': segres, 'postprocessed': segres}, axis=1)
        
        self.loss = F.mse_loss
        
    def save(self, notion):
        save_path = f'../saved/{notion}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)     
        np.save(f'{save_path}{self.name}_resized', self.resized['vanilla'])
        np.save(f'{save_path}{self.name}_resized-pp', self.resized['postprocessed'])
        np.save(f'{save_path}{self.name}_matched', self.matched['vanilla'])
        np.save(f'{save_path}{self.name}_matched-pp', self.matched['postprocessed'])
        np.save(f'{save_path}{self.name}_center', self.center['vanilla'])
        np.save(f'{save_path}{self.name}_center-pp', self.center['postprocessed'])

    def resize_loss(self):
        print(f"{self.name} resizing & calculating loss...")
        source = 'vanilla' if not self.postprocessed else 'postprocessed'
        for (pred_ind, take, group, segment, sample) in tqdm(self.tags.itertuples()):
            # Find gt sample from pred tag
            gt_ind = self.gt_tags.loc[(self.gt_tags['take']==take) & (self.gt_tags['sample']==sample)].index.values[0]

            self.resized[source][pred_ind] = cv2.resize(np.squeeze(self.img_pred[pred_ind]),
                                                (self.image_size[1], self.image_size[0]))
            self.result.loc[pred_ind, (source, 'mse')] = F.mse_loss(torch.from_numpy(self.resized[source][pred_ind]), 
                                                                    torch.from_numpy(self.gt[gt_ind])).numpy()
            if not self.postprocessed:
                # Find center
                im = self.resized[source][pred_ind]
                (T, timg) = cv2.threshold((np.squeeze(im) * 255).astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(timg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) != 0:
                    contour = max(contours, key=lambda x: cv2.contourArea(x))
                    area = cv2.contourArea(contour)

                    if area > 0:
                        x, y, w, h = cv2.boundingRect(contour)
                        self.center[source][pred_ind] = np.array([int(x + w / 2), int(y + h / 2)])                                
        if self.result.loc[:, (source, 'mse')].isnull().values.any():
            print("nan detected! ", end='')
        print("Done!")

    def calculate_cdf(self, show_fig=True):
        print(f"{self.name} calculating histograms...", end='')
        source = 'vanilla' if not self.postprocessed else 'postprocessed'
        hist, bin_edges = np.histogram(self.result.loc[:, (source, 'mse')].values)
        
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
                
    def post_processing(self, window_size=7, forced=False):
        if forced:
            print(f"{self.name} post processing...", end='')
            for (_, take, group, segment, samples, inds) in self.seg_tags.itertuples():
                selected = inds
                        
                # Only smooth x
                try:
                    selected_center = self.center['vanilla'][selected][..., 0]
                    processed_center = np.convolve(selected_center, np.ones(window_size) / window_size, mode='valid')

                    for pred_ind, center in zip(selected, processed_center):
                        self.center['postprocessed'][pred_ind][0] = center
                except Exception:
                    print(f'Skipped {self.name}')
            self.postprocessed = True
            print('Done!')
    
    def matching_mae(self, scale=0.3):
        print(f"{self.name} calculating 2D correlation...", end='')
        source = 'vanilla' if not self.postprocessed else 'postprocessed'
        for (pred_ind, take, group, segment, sample) in tqdm(self.tags.itertuples()):
            im = self.resized[source][pred_ind]
            
            _im = Image.fromarray((im * 255).astype(np.uint8))
            im = _im.resize((int(self.image_size[1]*scale), int(self.image_size[0]*scale)),Image.BILINEAR)
            im = np.array(im).astype(float)
            
            gt_ind = self.gt_tags.loc[(self.gt_tags['take']==take) & (self.gt_tags['sample']==sample)].index.values[0]
            gt = np.squeeze(self.gt[gt_ind])
            
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
            
            self.matched[source][pred_ind] = im_re
            self.result.loc[pred_ind, (source, ['matched_mae', 'dev_x', 'dev_y', 'deviation'])] = [np.mean(np.abs(gt_re - im_re)), x, y, np.sqrt(x**2+y**2)]

        print("Done!")
        
    def segment_mean(self):
        print(f"{self.name} calculating segment mean...", end='')
        source = 'vanilla' if not self.postprocessed else 'postprocessed'
        
        for (_, take, group, segment, samples, inds) in self.seg_tags.itertuples():
            for key in self.result[source].columns:
                self.seg_result.loc[_, ('segments', ['take', 'group', 'segment'])] = [take, group, segment]
                self.seg_result.loc[_, (source, key)] = np.mean(self.result.loc[inds, (source, key)])
        print('Done')
        
    def plot_example(self, selected=None, matched=False, source='vanilla'):
        fig = plot_settings()
        fig.suptitle(f"{self.name} Reconstruction Examples")
        filename = f"{self.name}_Reconstruct.jpg"

        plot_terms = {'Raw Ground Truth': self.gt,
                      'Raw Estimates': self.resized[source]}
        if matched:
            plot_terms['Matched Estimates'] = self.matched[source]
            filename = f"{self.name}_Reconstruct_Matched.jpg"
        subfigs = fig.subfigures(nrows=len(list(plot_terms.keys())), ncols=1)

        if not selected:
            selected = self.tags.sample(n=8)

        for i, (key, value) in enumerate(plot_terms.items()):
            subfigs[i].suptitle(key, fontweight="bold")
            axes = subfigs[i].subplots(nrows=1, ncols=len(selected))
            for (pred_ind, take, group, segment, sample), ax in zip(selected.itertuples(name=None), axes):
                gt_ind = self.gt_tags.loc[(self.gt_tags['take']==take) & (self.gt_tags['sample']==sample)].index.values[0]
                
                ax.imshow(np.squeeze(self.gt[gt_ind]) if key == 'Raw Ground Truth'
                               else np.squeeze(value[pred_ind]), vmin=0, vmax=1)
                if key == 'Matched Estimates':
                    st = [self.center[source][pred_ind][..., 0], 
                          self.center[source][pred_ind][..., 1]]
                    ed = [st[0] + self.result.loc[pred_ind, (source, 'dev_x')], 
                          st[1] + self.result.loc[pred_ind, (source, 'dev_y')]]
                    ax.plot([st[0], ed[0]], [st[1], ed[1]], color='red', linewidth=2)
                    
                ax.axis('off')
                ax.set_title(f"{'-'.join(map(str, (take, group, segment, sample)))}")
                ax += 1
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

    def resize_loss(self):
        self.fail_count = 0
        self.fail_ind = []
        print(f"{self.name} reconstructing...")
        source = 'vanilla' if not self.postprocessed else 'postprocessed'
        for (pred_ind, take, group, segment, sample) in tqdm(self.tags.itertuples()）:

            gt_ind = self.gt_tags.loc[(self.gt_tags['take']==take) & (self.gt_tags['sample']==sample)].index.values[0]
            img = np.squeeze(self.preds['S_PRED'][pred_ind]) * np.squeeze(self.depth[pred_ind])
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

                    x1, y1, x2, y2 = self.bbx[pred_ind]
                    
                    x0 = int(min(x1, x2) * 226)
                    y0 = int(min(y1, y2) * 128)
                    w0 = int(abs((x2 - x1) * 226))
                    h0 = int(abs((y2 - y1) * 128))

                    try:
                        subject1 = cv2.resize(subject, (w0, h0))
                        for x_ in range(w0):
                            for y_ in range(h0):
                                self.resized[source][pred_ind, y0 + y_, x0 + x_] = subject1[y_, x_]
                                self.result.loc[pred_ind, (source, 'mse')] = self.loss(torch.from_numpy(self.resized[source][pred_ind]), 
                                                                              torch.from_numpy(self.gt[gt_ind])).numpy()

                    except Exception as e:
                        print(e)
                        print(x0, y0, w0, h0)
                        self.fail_count += 1
                        self.fail_ind.append(pred_ind)
        if self.result[source]['mse'].isnull().values.any():
            print("nan detected! ", end='')
        print(f"Done\n Reconstruction finished. Failure count = {self.fail_count}")

    def plot_example(self, selected=None, matched=False, source='vanilla'):
        fig = plot_settings()
        fig.suptitle(f"{self.name} Reconstruction Examples")
        filename = f"{self.name}_Reconstruct.jpg"
       
        plot_terms = {'Cropped Ground Truth': self.preds['GT'],
                      'Cropped Estimates': self.preds['S_PRED'],
                      'Raw Ground Truth': self.gt,
                      'Raw Estimates': self.resized[source]}
        if matched:
            plot_terms['Matched Estimates'] = self.matched[source]
            filename = f"{self.name}_Reconstruct_Matched.jpg"
        subfigs = fig.subfigures(nrows=len(list(plot_terms.keys())), ncols=1)

        if not selected:
            selected = self.tags.sample(n=8)

        for i, (key, value) in enumerate(plot_terms.items()):
            subfigs[i].suptitle(key, fontweight="bold")
            axes = subfigs[i].subplots(nrows=1, ncols=len(selected))

            for (pred_ind, take, group, segment, sample), ax in zip(selected.itertuples(name=None), axes):
                gt_ind = self.gt_tags.loc[(self.gt_tags['take']==take) & (self.gt_tags['sample']==sample)].index.values[0]
                
                ax.imshow(np.squeeze(self.gt[gt_ind]) if key == 'Raw Ground Truth'
                                     else np.squeeze(value[pred_ind]), vmin=0, vmax=1)
                if key == 'Raw Ground Truth':
                    x1, y1, x2, y2 = self.preds['GT_BBX'][pred_ind]
                    x = int(x1 * 226)
                    y = int(y1 * 128)
                    w = int((x2 - x1) * 226)
                    h = int((y2 - y1) * 128)
                    ax.add_patch(Rectangle((x, y), w, h, edgecolor='pink', fill=False, lw=3))
                elif key == 'Raw Estimates':
                    x1, y1, x2, y2 = self.preds['S_BBX'][pred_ind]
                    x = int(x1 * 226)
                    y = int(y1 * 128)
                    w = int((x2 - x1) * 226)
                    h = int((y2 - y1) * 128)
                    ax.add_patch(Rectangle((x, y), w, h, edgecolor='orange', fill=False, lw=3))
                elif key == 'Matched Estimates':
                    st = [self.center[source][pred_ind][..., 0], 
                          self.center[source][pred_ind][..., 1]]
                    ed = [st[0] + self.result.loc[pred_ind, (source, 'dev_x')].values[0], 
                          st[1] + self.result.loc[pred_ind, (source, 'dev_y')].values[0]]
                    ax.plot([st[0], ed[0]], [st[1], ed[1]], color='red', linewidth=2)
                    
                ax.axis('off')
                ax.set_title(f"{'-'.join(map(str, (take, group, segment, sample)))}")
        plt.show()
        
        return {filename: fig}


class CenterResultCalculator(ResultCalculator):
    def __init__(self, *args, **kwargs):
        super(CenterResultCalculator, self).__init__(*args, **kwargs)

        self.depth = np.array(self.preds['S_DPT'])

        self.fail_count = 0
        self.fail_ind = []

    def resize_loss(self):
        self.fail_count = 0
        self.fail_ind = []
        print(f"{self.name} reconstructing...")
        source = 'vanilla' if not self.postprocesed else 'postprocessed'
        for (pred_ind, take, group, segment, sample) in tqdm(self.tags.itertuples()):
            gt_ind = self.gt_tags.loc[(self.gt_tags['take']==take) & (self.gt_tags['sample']==sample)].index.values[0]
            
            x, y = self.preds['S_CTR'][pred_ind]
            x, y = int(x * 226), int(y * 226)

            # Pad cropped image into raw-image size
            img = np.squeeze(self.img_pred[pred_ind]) * self.depth[pred_ind]
            new_img = np.pad(img, [(0, 0), (49, 49)], 'constant', constant_values=0)

            try:
                # Roll to estimated position
                new_img = np.roll(new_img, y-64, axis=0)
                new_img = np.roll(new_img, x-113, axis=1)
                self.resized[source][pred_ind] = new_img
                self.result.loc[pred_ind, (source, 'mse')] = F.mse_loss(torch.from_numpy(self.resized[source][pred_ind]), 
                                                                        torch.from_numpy(self.gt[gt_ind])).numpy()

            except Exception as e:
                print(e)
                print(x, y)
                self.fail_count += 1
                self.fail_ind.append(pred_ind)

        if self.result[source]['mse'].isnull().values.any():
            print("nan detected! ", end='')
        print(f"Done\n Reconstruction finished. Failure count = {self.fail_count}")

    def plot_example(self, selected=None, matched=False, source='vanilla'):
        fig = plot_settings()
        fig.suptitle(f"{self.name} Reconstruction Examples")
        filename = f"{self.name}_Reconstruct.jpg"

        plot_terms = {
            'Cropped Ground Truth': self.preds['GT'],
            'Cropped Estimates'   : self.preds['S_PRED'],
            'Raw Ground Truth'    : self.gt,
            'Raw Estimates'       : self.resized[source]}
        if matched:
            plot_terms['Matched Estimates'] = self.matched[source]
            filename = f"{self.name}_Reconstruct_Matched.jpg"
            
        subfigs = fig.subfigures(nrows=len(list(plot_terms.keys())), ncols=1)
        
        if not selected:
            selected = self.tags.sample(n=8)
            
        for i, (key, value) in enumerate(plot_terms.items()):
            subfigs[i].suptitle(key, fontweight="bold")
            axes = subfigs[i].subplots(nrows=1, ncols=len(selected))
            for (pred_ind, take, group, segment, sample), ax in zip(selected.itertuples(name=None), axes):
                gt_ind = self.gt_tags.loc[(self.gt_tags['take']==take) & (self.gt_tags['sample']==sample)].index.values[0]
                
                ax.imshow(np.squeeze(self.gt[gt_ind]) if key == 'Raw Ground Truth'
                               else np.squeeze(value[pred_ind]), vmin=0, vmax=1)

                if key == 'Raw Ground Truth':
                    x, y= self.preds['GT_CTR'][pred_ind]
                    x = int(x * 226)
                    y = int(y * 128)
                    ax.scatter(x, y, c='red', marker=(5, 1), alpha=0.5, linewidths=5, label='GT_CTR')
                elif key == 'Raw Estimates':
                    x, y= self.preds['S_CTR'][pred_ind]
                    x = int(x * 226)
                    y = int(y * 128)
                    ax.scatter(x, y, c='red', marker=(5, 1), alpha=0.5, linewidths=5, label='S_CTR')
                elif key == 'Matched Estimates':
                    st = [self.center[source][pred_ind][..., 0], 
                          self.center[source][pred_ind][..., 1]]
                    ed = [st[0] + self.result.loc[pred_ind, (source, 'dev_x')].values[0], 
                          st[1] + self.result.loc[pred_ind, (source, 'dev_y')].values[0]]
                    ax.plot([st[0], ed[0]], [st[1], ed[1]], color='red', linewidth=2)
                    
                ax.axis('off')
                ax.set_title(f"{'-'.join(map(str, (take, group, segment, sample)))}")
        plt.show()

        return {filename: fig}
    
    def post_processing(self, window_size=7, *args, **kwargs):
        print(f"{self.name} post processing...", end='')
        for (_, take, group, segment, samples, inds) in self.seg_tags.itertuples():
            selected = inds
                    
            # Only smooth x
            try:
                selected_center = self.preds['S_CTR'][selected][..., 0]
                processed_center = np.convolve(selected_center, np.ones(window_size) / window_size, mode='valid')

                for pred_ind, center in zip(selected, processed_center):
                    self.preds.center['postprocessed'][pred_ind][0] = center
            except Exception:
                print(f'Skipped {self.name}')
        self.postprocessed = True
        print('Done!')

class ZeroEstimates(ResultCalculator):
    zero = True

    def __init__(self, *args, **kwargs):
        super(ZeroEstimates, self).__init__(*args, **kwargs)

        print(f"{self.name} generated Zero Estimates")
        self.len = len(self.gt)
        self.tags, self.seg_tags = self.structurize_tags(self.gt_tags)
        
    def resize_loss(self):
        print(f"{self.name} resized")
        source = 'vanilla' if not self.postprocessed else 'postprocessed'
        for (gt_ind, take, group, segment, sample) in self.tags.itertuples():
            self.result.loc[gt_ind, (source, 'mse')] = F.mse_loss(torch.from_numpy(self.resized[source][gt_ind]), 
                                                                  torch.from_numpy(self.gt[gt_ind])).numpy()

    def find_center(self, *args, **kwargs):
        pass
    
    def matching_mae(self, *args, **kwargs):
        print(f"{self.name} calculating 2D correlation...", end='')
        source = 'vanilla' if not self.postprocessed else 'postprocessed'
        for (gt_ind, take, group, segment, sample) in self.tags.itertuples():
            self.result.loc[gt_ind, (source, ['matched_mae', 'dev_x', 'dev_y', 'deviation'])] = [
                np.mean(np.abs(np.squeeze(self.gt[gt_ind]) - self.resized[source][gt_ind])), 
                self.image_size[1], 
                self.image_size[0], 
                np.sqrt(self.image_size[1]**2+self.image_size[0]**2)]
        print("Done!")

class GatherPlotCDF:
    def __init__(self, subjects:dict):
        self.subjects = subjects
        self.count = 0
        self.cdf: dict = {}
        
    def __call__(self, scope='all', customize=False, item='mse', level='sample', source='vanilla'):
        scope = list(self.subjects.keys()) if scope=='all' else scope
        fig = plot_settings((20, 10))
        if not customize:
            filename = f"Comparison_CDF_{item.upper()}_{level}_{source}_{self.count}.jpg"
            fig.suptitle(f'Test PDF-CDF - {item.upper()} - {level} - {source}', fontweight="bold")
        else:
            filename = f"Comparison_CDF_{customize}_{self.count}.jpg"
            fig.suptitle(f'Test PDF-CDF - {customize}', fontweight="bold")
       
        _mmax = 0
        lev = 'result' if level=='sample' else 'seg_result'
        self.cdf[item] = {}
        
        for i, sub in enumerate(scope):
            # sub = [sub, item, level, source]
            if customize:
                sub, item, level, source = sub
            mmin = [np.min(getattr(s, lev)[source][item].values) for s in self.subjects.values()]
            mmax = [np.max(getattr(s, lev)[source][item].values) for s in self.subjects.values()]
            
            nbins = 50 if level=='sample' else 25
            bins = np.linspace(np.min(mmin), np.max(mmax), nbins)
            hist_, bin_edges = np.histogram(getattr(self.subjects[sub], lev)[source][item].values, bins)

            width = (bin_edges[1] - bin_edges[0]) * 0.8
            cdf = np.cumsum(hist_ / sum(hist_))           
            _mmax = np.max(bin_edges) if np.max(bin_edges) > _mmax else _mmax
            self.cdf[item][sub] = cdf
            if not self.subjects[sub].zero:
                plt.bar(bin_edges[1:], hist_ / max(hist_), width=width, label=sub, zorder=i)
                plt.plot(bin_edges[1:], cdf, '-*', label=sub, zorder=i + len(scope))
            else:
                plt.bar(bin_edges[1:], hist_ / max(hist_), width=width * 0.4, label=sub, zorder=len(scope))
                plt.plot(bin_edges[1:], cdf, '-.', marker='o', label=sub, zorder=2 * len(scope))

        ax = plt.gca()
        ax.fill_between(np.arange(0, _mmax, 0.01), 1.02,
                        color='white', alpha=0.5, zorder=len(scope) + 0.5)

        plt.xlabel(f'Per-{level} {item}')
        plt.ylabel('Frequency')
        plt.grid()
        plt.legend()
        plt.show()
        self.count += 1

        return {filename: fig}
    
    def print_table(self):
        return pd.DataFrame(self.cdf, columns=list(range(49)))
    
class GatherPlotBox:
    def __init__(self, subjects:dict):
        self.subjects = subjects
        self.count = 0
        
    def __call__(self, scope='all', customize=False, item='mse', level='sample', source='vanilla'):
        scope = list(self.subjects.keys()) if scope=='all' else scope
        lev = 'result' if level=='sample' else 'seg_result'
        fig = plot_settings((20, 10))
        if not customize:
            filename = f"Comparison_BoxPlot_{item.upper()}_{level}_{source}_{self.count}.jpg"
            fig.suptitle(f'Test Box Plot - {item.upper()} - {level} - {source}', fontweight="bold")
        else:
            filename = f"Comparison_BoxPlot_{customize}_{self.count}.jpg"
            fig.suptitle(f'Test Box Plot - {customize}', fontweight="bold")
        
        
        for i, sub in enumerate(scope):
            # sub = [sub, item, level, source]
            if customize:
                sub, item, level, source = sub
            plt.boxplot(getattr(self.subjects[sub], lev)[source][item].values, labels=[sub], positions=[i+1], vert=True, showmeans=True,
            patch_artist=True,
            boxprops={'facecolor': 'lightblue'})
        plt.yscale('log', base=2)
        plt.show()
        self.count += 1
        return {filename: fig}

class ResultVisualize:
    def __init__(self, subjects:dict, univ_gt):
        self.subjects = subjects
        self.univ_gt = univ_gt['rimg']
        self.univ_tag, self.univ_seg_tag = Estimates.structurize_tags(univ_gt['tag'])

        self.selected = None
        self.seg_selected = None
        # 250-pixel height for each row
        self.figsize = (20, 2.5 * (len(self.subjects) + 1))
        self.count = 0
        
    def __call__(self, scope='all', customize=False, selected=None, matched=False, level='sample', source='vanilla'):
        scope = list(self.subjects.keys()) if scope=='all' else scope
        
        fig = plot_settings(self.figsize)
        mch = ' - Matched' if matched else ''
        if not customize:
            fig.suptitle(f'Comparison Visualization{mch} - {level} - {source}')
            filename = f"Comparison_Visual_Matched_{self.count}.jpg" if matched else f"Comparison_Visual_{self.count}.jpg"
        else:
            fig.suptitle(f'Comparison Visualization{mch} - {customize}')
            filename = f"Comparison_Visual_Matched_{customize}_{self.count}.jpg" if matched else f"Comparison_Visual_{self.count}.jpg"
       
        if level == 'sample':
            # Randomly pick 8 samples
            if selected is None:
                if self.selected is None:
                    selected = self.univ_tag.sample(n=8)
                    self.selected = selected
                else:
                    selected = self.selected

        else:
            #Randomly pick a segment, 8 sequential samples
            if self.seg_selected is None:
                selected = self.univ_seg_tag.sample(n=1)
                selected = selected.explode('inds')
                selected = selected[['inds', 'take', 'group', 'segment', 'samples']]
                for i, sample in enumerate(selected.samples.tolist()):
                    selected.loc[i, 'samples'] = sample
                
                if len(selected) >= 8:
                    selected = selected[:8]
                self.seg_selected = selected
            else:
                selected = self.seg_selected
        
        ncols = len(selected.inds.tolist())
        subfigs = fig.subfigures(nrows=len(scope) + 1, ncols=1)
        subfigs[0].suptitle("Ground Truth", fontweight="bold")
        axes = subfigs[0].subplots(nrows=1, ncols=ncols)
        for (_, gt_ind, take, group, segment, sample), ax in zip(selected.itertuples(name=None), axes):
            ax.imshow(np.squeeze(self.univ_gt[gt_ind]), vmin=0, vmax=1)
            ax.axis('off')
            ax.set_title(f"{'-'.join(map(str, (take, group, segment, sample)))}")

        for i, sub in enumerate(scope):
            # sub = [sub, level, source]
            if customize:
                sub, level, source = sub
            if not self.subjects[sub].zero:
                pp = '' if not self.subjects[sub].postprocessed else '-pp'
                subfigs[i + 1].suptitle(f'{self.subjects[sub].name}{pp}', fontweight="bold")
                axes = subfigs[i + 1].subplots(nrows=1, ncols=ncols)
                for (_, gt_ind, take, group, segment, sample), ax in zip(selected.itertuples(name=None), axes):
                    pred_ind = self.subjects[sub].tags[(self.subjects[sub].tags['take']==take) & 
                                                       (self.subjects[sub].tags['sample']==sample)].index.values[0]
                    ax.imshow(np.squeeze(self.subjects[sub].matched[source][pred_ind] if matched 
                                         else self.subjects[sub].resized[source][pred_ind]), vmin=0, vmax=1)
                    ax.axis('off')
                    ax.set_title(f"{'-'.join(map(str, (take, group, segment, sample)))}")
                    if matched:
                        st = [self.subjects[sub].center[source][pred_ind][..., 0], 
                            self.subjects[sub].center[source][pred_ind][..., 1]]
                        ed = [st[0] + self.subjects[sub].result.loc[pred_ind, (source, 'dev_x')], 
                            st[1] + self.subjects[sub].result.loc[pred_ind, (source, 'dev_y')]]
                        ax.plot([st[0], ed[0]], [st[1], ed[1]], color='red', linewidth=2)
        plt.show()
        self.count += 1
        return {filename: fig}
    
class ResultProcess:
    def __init__(self, subjects:dict, most_gt, least_gt):
        self.subjects = subjects
        self.most_gt = most_gt
        self.least_gt = least_gt
        self._cdfplot = GatherPlotCDF(subjects=self.subjects)
        self._boxplot = GatherPlotBox(subjects=self.subjects)
        self._visualization = ResultVisualize(subjects=self.subjects, univ_gt=self.least_gt)
        self.figs: dict = {}
        self.table = None
        
    def load_preds(self):
        for sub, path in self.subjects.items():
            print(f'Loading {sub}...')
            if 'Center' in sub:
                self.subjects[sub] = CenterResultCalculator(name=sub, path=path, gt=self.most_gt)
            elif 'BBX' in sub:
                self.subjects[sub] = BBXResultCalculator(name=sub, path=path, gt=self.most_gt)
            elif 'Zero' in sub:
                self.subjects[sub] = ZeroEstimates(name=sub, path=None, gt=self.most_gt)
            else:
                self.subjects[sub] = ResultCalculator(name=sub, path=path, gt=self.most_gt)
                
    def resize(self, scope='all', show_fig=False, **kwargs):
        scope = list(self.subjects.keys()) if scope=='all' else scope
        for sub in scope:
            self.subjects[sub].resize_loss()
            if show_fig:
                self.figs.update(self.subjects[sub].plot_example(**kwargs))
                
    def matching_mae(self, scope='all', scale=0.3, show_fig=False, **kwargs):
        scope = list(self.subjects.keys()) if scope=='all' else scope
        for sub in scope:
            self.subjects[sub].matching_mae(scale=scale)
            if show_fig:
                self.figs.update(self.subjects[sub].plot_example(matched=True, **kwargs))
                
    def post_process(self, scope='all', window_size=7, show_fig=False, **kwargs):
        scope = list(self.subjects.keys()) if scope=='all' else scope
        for sub in scope:
            self.subjects[sub].post_processing(window_size)
            self.subjects[sub].resize_loss()
            self.subjects[sub].matching_mae()
            if show_fig:
                self.figs.update(self.subjects[sub].plot_example(matched=True, **kwargs))
                
    def segment_mean(self, scope='all'):
        scope = list(self.subjects.keys()) if scope=='all' else scope
        for sub in scope:
            self.subjects[sub].segment_mean()
            
    def average_table(self, scope='all'):
        scope = list(self.subjects.keys()) if scope=='all' else scope
        keys = ['mse', 'matched_mae', 'dev_x', 'dev_y', 'deviation']
        vanilla_sample = pd.DataFrame(columns=keys)
        vanilla_segment = pd.DataFrame(columns=keys)
        postprocessed_sample = pd.DataFrame(columns=keys)
        postprocessed_segment = pd.DataFrame(columns=keys)
                
        for sub in scope:
            for key in keys:
                vanilla_sample.loc[sub, key] = np.mean(self.subjects[sub].result['vanilla'][key].values)
                vanilla_segment.loc[sub, key] = np.mean(self.subjects[sub].seg_result['vanilla'][key].values)
                postprocessed_sample.loc[sub, key] = np.mean(self.subjects[sub].result['postprocessed'][key].values)
                postprocessed_segment.loc[sub, key] = np.mean(self.subjects[sub].seg_result['postprocessed'][key].values)
        
        vanilla = pd.concat({'sample': vanilla_sample, 'segment': vanilla_segment}, axis=1)
        postprocessed = pd.concat({'sample': postprocessed_sample, 'segment': postprocessed_segment}, axis=1)
        table = pd.concat({'vanilla': vanilla, 'postprocessed': postprocessed}, axis=1)
        self.table = table
        print(table)
            
    def cdfplot(self, *args, **kwargs):
        self.figs.update(self._cdfplot(*args, **kwargs))
        
    def boxplot(self, *args, **kwargs):
        self.figs.update(self._boxplot(*args, **kwargs))
        
    def visualize(self, *args, **kwargs):
        self.figs.update(self._visualization(*args, **kwargs))
        
    def schedule(self):
        self.resize()
        self.matching_mae()
        self.segment_mean()
        for item in ('mse', 'matched_mae', 'deviation'):
            for level in ('sample', 'segment'):
                self.cdfplot(item=item, level=level, source='vanilla')
                self.boxplot(item=item, level=level, source='vanilla')
        self.visualize(matched=False, level='segment', source='vanilla')
        self.visualize(matched=True, level='segment', source='vanilla')
        self.post_process()
        self.segment_mean()
        for item in ('mse', 'matched_mae', 'deviation'):
            for level in ('sample', 'segment'):
                self.cdfplot(item=item, level=level, source='postprocessed')
                self.boxplot(item=item, level=level, source='postprocessed')
        self.visualize(matched=False, level='segment', source='postprocessed')
        self.visualize(matched=True, level='segment', source='postprocessed')
        
    def save_figs(self, notion):
        save_path = f'../saved/{notion}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for filename, fig in self.figs.items():
            fig.savefig(f"{save_path}{filename}")