import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import pandas as pd
from tqdm.notebook import tqdm

import cupy as cp
import cupyx.scipy.ndimage as cnd
import cupyx.scipy.signal as cps

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from misc import plot_settings
from PIL import Image
from scipy.ndimage import zoom

criteria = ['mse', 'matched_mae', 'matched_mae_mask', 'average_depth', 'est_depth', 'gt_depth', 'distance', 'x', 'y']


class Tester:
    
    # Notice: set shuffle=False for test loader!
    # Test on all test samples
    def __init__(self, name, trainer, total_length, save_path=None):
        self.name = name
        self.trainer = trainer
        self.preds = None
        self.total_length = total_length  # Same as Labels
        self.save_path = save_path
        
    def test(self, save=False, pred='R_PRED'):
        self.trainer.test(loader='test')
        if save:
            self.trainer.losslog.save('preds', self.save_path)
        
        self.preds = pd.DataFrame(index=list(range[self.total_length]), columns=['gt', 'pred', 'tag', 'matched', 'center'])
        
        gt = self.trainer.losslog.preds['R_GT']
        preds = self.trainer.losslog.preds[pred]
        tags = self.trainer.losslog.preds['TAG']
        abs_ind = self.trainer.losslog.preds['IND']
        
        # Store results by absolute indicies
        for i, ind in enumerate(abs_ind):
            self.preds.loc[ind, ['gt', 'pred', 'tag']] = [gt[i], preds[i], tags[i]]


class ResultCalculator(Tester):
    def __init__(self, *args, **kwargs):
        super(ResultCalculator, self).__init__(*args, **kwargs)
        self.image_size = (128, 128)
    
    @staticmethod
    def find_center(img):
        (T, timg) = cv2.threshold((np.squeeze(img) * 255).astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(timg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 0:
            contour = max(contours, key=lambda x: cv2.contourArea(x))
            area = cv2.contourArea(contour)
            if area > 0:
                x, y, w, h = cv2.boundingRect(contour)
                return np.array([x + w / 2, y + h / 2]).astype(int)
        else:
            return np.zeros((1, 4))
        
    @staticmethod
    def matched_mae(im, gt, scale, cuda): 
        # threshold 0.1 => 36cm
        im_mask = np.where(im_re > 0.1, 1., 0.)
        gt_mask = np.where(gt > 0.1, 1., 0.)
        
        im_depth = im.sum() / im_mask.sum()
        gt_depth = gt.sum() / gt_mask.sum()
        average_depth = np.abs(gt_depth - im_depth)
        
        new_size = int(128*scale), int(128*scale)
        
        _im = Image.fromarray((im * 255).astype(np.uint8))
        _im = _im.resize(new_size, Image.BILINEAR)
        _im = np.array(_im).astype(float)
        
        _gt = Image.fromarray((gt * 255).astype(np.uint8))
        _gt = _gt.resize(new_size, Image.BILINEAR)
        _gt = np.array(_gt).astype(float)
    
        tmp = cps.correlate2d(cp.array(gt), cp.array(im), mode='full')
        tmp = cp.asnumpy(tmp)
        
        y, x = np.unravel_index(np.argmax(tmp), tmp.shape)
        y = int(y / scale) - 128
        x = int(x / scale) - 128
    
        im_re = im.rotate(0, translate = (x, y))
        im_re = np.array(im_re).astype(float) / 255.
        
        matched_mae = np.mean(np.abs(gt - im_re))
        matched_mae_mask = np.mean(np.abs(im_mask - gt_mask))
        
        distance = np.sqrt(x**2+y**2)

        return im_re, average_depth, matched_mae, matched_mae_mask, distance, x, y
    
    def mse(self, im, gt, find_center=False):
        mse = F.mse_loss(torch.from_numpy(im), torch.from_numpy(gt)).numpy()
        if find_center:
            center = self.find_center(im)
            mse = (mse, center)
        return mse
        
    def evaluate(self, scale=3, cuda=0, find_center=False):
        self.results = pd.DataFrame(index=list(range[self.total_length]), columns=criteria)
        
        for i, gt, pred, *_ in tqdm(self.preds.itertuples(), desc="Evaluating"):
            
            mse = self.mse(pred, gt)
            im_re, average_depth, matched_mae, matched_mae_mask, distance = self.matched_mae(pred, gt, scale, cuda)
            
            self.results.loc[i, ['mse', 'matched_mae', 'matched_mae_mask', 'average_depth', 'distance', 'x', 'y']] = mse, 
            matched_mae, matched_mae_mask, average_depth, distance
            self.preds.loc[i, 'matched'] = im_re
            
    def plot_example(self, selected=None, matched=False, source='vanilla'):
        fig = plot_settings()
        fig.suptitle(f"{self.name} Depth Image Estimates")
        filename = f"{self.name}_Depth_Image.jpg"

        plot_terms = {'Ground Truth': self.preds['gt'],
                      'Estimates': self.preds['pred']}
        if matched:
            plot_terms['Matched Estimates'] = self.preds['matched']
            filename = f"{self.name}_Depth_Image_Matched.jpg"
        subfigs = fig.subfigures(nrows=len(list(plot_terms.keys())), ncols=1)

        if not selected:
            selected = np.choice(np.arange(self.len), 8)

        zoom_factors = (1, 226 / 128) 

        for i, (key, value) in enumerate(plot_terms.items()):
            subfigs[i].suptitle(key, fontweight="bold")
            axes = subfigs[i].subplots(nrows=1, ncols=len(selected))
            for ind, ax in zip(selected, axes):
                ax.imshow(zoom(value[ind].squeeze(), zoom_factors), vmin=0, vmax=1)
                
                if key == 'Matched Estimates':
                    st = self.preds.loc[ind, 'center'], 
                    ed = [st[0] + self.result.loc[ind, 'x'], 
                          st[1] + self.result.loc[ind, 'y']]
                    ax.plot([st[0], ed[0]], [st[1], ed[1]], color='red', linewidth=2)
                    
                ax.axis('off')
                ax.set_title(self.tags[ind])

        plt.show()
        return {filename: fig}
    
    
class ResultProcess:
    def __init__(self, subjects, labels):
        # subject = {name: trainer}
        # labels = total_segment_labels (as in DataOrganizer)
        self.subjects = subjects
        self.selected = None
        self.table = None
        self.vis_count = 0
        self.box_count = 0
        self.cdf_count = 0
        self.nsamples = 12
        self.figs: dict = {}
        
    def test(self):
        for sub, trainer in self.subjects.items():
            trainer.test()
            
    def visualization(self, scope='all', selected=None, matched=False):
        if selected is None:
            if self.selected is None:
                selected = self.univ_tag.sample(n=self.nsamples)
                self.selected = selected
            else:
                selected = self.selected
        
        scope = list(self.subjects.keys()) if scope=='all' else scope
        self.figsize = (2.5 * self.nsamples, 2.5 * (len(scope) + 1))
        fig = plot_settings(self.figsize)
        mch = ' - Matched' if matched else ''
        
        fig.suptitle(f'Comparison Visualization{mch}')
        filename = f"Comparison_Visual_Matched_{self.vis_count}.jpg" if matched else f"Comparison_Visual_{self.vis_count}.jpg"
        zoom_factors = (1, 226 / 128)
         
        subfigs = fig.subfigures(nrows=len(scope) + 1, ncols=1)
        
        # Show GT
        subfigs[0].suptitle("Ground Truth", fontweight="bold")
        axes = subfigs[0].subplots(nrows=1, ncols=self.nsamples)
        for ind, ax in zip(selected, axes):
            ax.imshow(self.subjets[scope[0]].gt[ind].squeeze(), vmin=0, vmax=1)
            ax.set_title(self.subjets[scope[0]].tags[ind])
            ax.axis('off')
            
        # Show pred
        for sub, subfig in zip(scope, subfigs[1:]):
            _sub = self.subjects[sub]

            subfig.suptitle(f'{_sub.name}', fontweight="bold")
            axes = subfig.subplots(nrows=1, ncols=self.nsamples)
            for ind, ax in zip(selected, axes):
                im = getattr(_sub, 'matched' if matched else 'resized')[ind].squeeze()
                ax.imshow(zoom(im, zoom_factors), vmin=0, vmax=1)
                ax.axis('off')
                ax.set_title(_sub.tags[ind])

                if matched:
                    st = _sub.center[ind][0]
                    ed = [st[0] + _sub.result.loc[ind, 'x'].values, 
                        st[1] + _sub.result.loc[ind, 'y'].values]
                    ax.plot([st[0], ed[0]], [st[1], ed[1]], color='red', linewidth=2)
        plt.show()
        self.vis_count += 1
        self.figs[filename] = fig
    
    def gathercdf(self, scope='all', item='mse'):
        scope = list(self.subjects.keys()) if scope=='all' else scope
        fig = plot_settings((20, 10))

        filename = f"Comparison_CDF_{item.upper()}_{self.cdf_count}.jpg"
        fig.suptitle(f'Test PDF-CDF - {item.upper()}', fontweight="bold")
       
        _mmax = 0
        self.cdf[item] = {}
        
        for i, sub in enumerate(scope):
            mmin = [np.min(self.subjects[sub].results[item].values)]
            mmax = [np.max(self.subjects[sub].results[item].values)]
            
            nbins = 50
            bins = np.linspace(np.min(mmin), np.max(mmax), nbins)
            hist_, bin_edges = np.histogram(self.subjects[sub].results[item].values, bins)

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

        plt.xlabel(f'Per-sample {item}')
        plt.ylabel('Frequency')
        plt.grid()
        plt.legend()
        plt.show()
        self.cdf_count += 1
        self.figs[filename] = fig
    
    def gatherbox(self, scope='all', item='mse'):
        scope = list(self.subjects.keys()) if scope=='all' else scope
        fig = plot_settings((2*(len(scope) + 1) if len(scope)>9 else 20, 10))
 
        filename = f"Comparison_BoxPlot_{item.upper()}_{self.box_count}.jpg"
        fig.suptitle(f'Test Box Plot - {item.upper()}', fontweight="bold")
        
        for i, sub in enumerate(scope):
            _sub = self.subjects[sub]
            plt.boxplot(_sub.results[item].values, 
                        labels=[sub], positions=[i+1], vert=True, showmeans=True,
                        patch_artist=True, boxprops={'facecolor': 'lightblue'})
        plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')    
        plt.yscale('log', base=2)
        plt.show()
        self.box_count += 1
        self.figs[filename] = fig
    
    def average_table(self, scope='all'):
        scope = list(self.subjects.keys()) if scope=='all' else scope
        keys = criteria
        comparison_table = pd.DataFrame(columns=keys)
                
        for sub in scope:
            for key in keys:
                comparison_table.loc[sub, key] = np.mean(self.subjects[sub].results[key].values)

        self.table = comparison_table
        print(comparison_table)