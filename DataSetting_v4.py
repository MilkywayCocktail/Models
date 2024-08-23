import torch
import torch.utils.data as Data
import torch.distributed as dist
from torchvision import transforms

import numpy as np
import pandas as pd
from scipy import signal
import os
from PIL import Image

from tqdm.notebook import tqdm

class Raw:
    """
    Store raw data and avoid being changed
    """
    def __init__(self, value):
        self._value = value.copy()
        self._value.setflags(write=False)
        
    # Make sure to use copy() when assigning values!
    @property
    def value(self):
        return self._value
    
class LabelLoader:
    def __init__(self, label_path, *args, **kwargs):
        # CSI and IMG timestamps are in the same path
        # Load labels and pair samples by timestamps for each environment
        
        self.label_path = label_path
        self.labels: dict = {}
        
        self.csi_time: dict = {}
        self.img_time: dict = {}
        
        self.segment_labels = pd.DataFrame(columns=['env', 'bag', 'csi', 'subject', 'group', 'segment', 'img_inds', 'csi_inds'])
        
        self.alignment = 'tail'
        self.csi_len = 300

    def load_label(self):
        """
        Load labels at environment level (load all subjects)
        """
        print(f'Loading label...', end='')
        paths = os.walk(self.label_path)
        for path, _, file_lst in paths:
            for file_name in file_lst:
                file_name_, ext = os.path.splitext(file_name)
                
                # Load Label
                # Label filename is <subject.csv>
                if ext == '.csv':
                    self.labels[file_name_] = pd.read_csv(os.path.join(path, file_name))
                    # Adjust scale
                    self.labels[file_name_].loc[:, ['start', 'end', 'open', 'shut']] *= 1.e-3
                    
                # Load IMG time and CSI time
                if ext == '.txt':
                    if 'cam' in file_name:
                        self.img_time[file_name] = (np.load(os.path.join(path, file_name)))
                        
                    elif 'csi' in file_name:
                        self.csi_time[file_name] = (np.load(os.path.join(path, file_name)))
        print('Done')
        
    def save(self):
        for subject in self.labels.keys():
            sub_labels = self.segment_labels.loc[(self.segment_labels['subject']==subject)]
            sub_labels.to_csv(os.path.join(self.label_path, f'{subject}_matched.csv'))
        print('Matched labels saved!')
        
    def match(self):
        """
        Match CSI and IMG by timestamps
        """
        tqdm.write(f"{self.name} matching ...\n")
        for subject in tqdm(self.labels.keys()):
            for (env, bag, csi, group, segment, start, end, open, shut) in self.labels[subject].itertuples():
                start_img_id, end_img_id = np.searchsorted(self.img_time[bag], start, end)
                # Fill in-between ids
                img_id = np.arange(start_img_id, end_img_id, dtype=int) 

                # Pairing up CSI-IMG series
                csi_id = np.searchsorted(self.csi_time[csi], self.img_time[bag][img_id])
                
                # Align multiple CSI packets to one IMG
                # start_csi_id, end_csi_id = np.searchsorted(self.csi_time[subject], start, end)
                
                for i, (img_id_, csi_id_) in enumerate(zip(img_id, csi_id)):
                    if self.alignment == 'head':
                        sample_csi_id = np.arange(csi_id_, csi_id_ + self.csi_len, dtype=int)
                        
                    elif self.alignment == 'tail':
                        sample_csi_id = np.arange(csi_id_ - self.csi_len, csi_id_, dtype=int)
                        
                    # Remove out-of-segment sample
                    # if sample_csi_id[0] < start_csi_id or sample_csi_id[-1] > end_csi_id:
                    #     continue
                    
                    rec = [env, bag, csi, subject, group, segment, img_id_, csi_id_]
                    self.segment_labels[len(self.segment_labels)] = rec

        tqdm.write(' Done matching')
        

class FilterPD:
    def __init__(self, k=21):
        self.k = k
    
    def __call__(self, csi):
        def cal_pd(u):
            pd = u[:, 1:, 0] * u[:, :-1, 0].conj()
            md = np.zeros_like(pd)
            for i in range(md.shape[1]):
                md[:, i] = signal.medfilt(pd[:, i].real, self.k) + 1.j * signal.medfilt(pd[:, i].imag, self.k)
            
            return np.concatenate((md.real, md.imag), axis=-1).astype(np.float32)
        
        try:
            # CSI shape = 300 * 30 * 3
            # Reshape into 3 * (30 * 300)
            u, *_ =np.linalg.svd(csi.transpose(2, 1, 0).reshape(3, -1), full_matrices=False)
            # AoA = segment_len * 4 (real & imag of 2)
            aoa = cal_pd(u)
            
            # Reshape into 30 * (3 * 300)
            u, *_ =np.linalg.svd(csi.transpose(1, 2, 0).reshape(30, -1), full_matrices=False)
            # ToF = segment_len * 58 (real & imag of 29)
            tof = cal_pd(u)
            
            # Concatenate as a flattened vector
            pd = np.concatenate((aoa, tof), axis=-1)
            
        except Exception as e:
            print(f'FilterPD aborted due to {e}')
        
        return pd
    
class MyDataset(Data.Dataset):
    """
    DATASET wrapper
    Load CSI, IMG, IMG-related modalities (CIMG, DPT, CTR)
    Filter CSI and Calculate PhaseDiff on-the-fly
    """

    def __init__(self,
                 data,
                 transform,
                 *args, **kwargs):

        self.data = data
        self.transform = transform
        self.filterpd = FilterPD()

    def __getitem__(self, index):
        """
        On-the-fly: filter and reshape CSI, calculate phasediff, transform images
        """
        ret = {key: value[index] for key, value in self.data.items()}

        # <1> Filter and reshape CSI, calculate phasediff
        assert 'csi' in self.data.keys()
        # packet * sub * rx
        csi_real = signal.savgol_filter(np.real(ret['csi']), 21, 3, axis=0)  # denoise for real part
        csi_imag = signal.savgol_filter(np.imag(ret['csi']), 21, 3, axis=0)  # denoise for imag part
        csi_complex = csi_real + 1.j * csi_imag
        
        # packet * sub * (rx * 2)
        csi = np.concatenate((csi_real, csi_imag), axis=-1)
        
        # (rx * 2) * sub * packet
        csi = csi.transpose(2, 1, 0)
        
        ret['csi'] = csi
        ret['pd'] = self.filterpd(csi_complex)
        
        # <2> Transform images
        if self.transform:
            if 'rimg' in ret.keys():
                ret['rimg'] = self.transform(Image.fromarray(np.squeeze(np.uint8(np.asarray(ret['rimg']) * 255))))
            if 'cimg' in ret.keys():
                ret['cimg'] = self.transform(Image.fromarray(np.squeeze(np.uint8(np.asarray(ret['cimg']) * 255))))
        
        return ret

    def __len__(self):
        return list(self.data.values())[0].shape[0]

class CrossValidator:
    """
    Generate labels for cross validation
    """
    def __init__(self, labels, level):
        self.labels = labels
        self.level = level
        self.current = 0
        
        self.range = {'A308T'} if self.level == 'day' else set(self.labels.loc[:, level].values)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= len(self.range):
            raise StopIteration
            
        else:
            if self.level == 'day':
                train_labels = self.labels[self.labels['env']=='A308']
                test_labels = self.labels[self.labels['env']=='A308T']
            else:
                # Select one as leave-1-out test
                train_labels = self.labels[self.labels[self.level]!=self.range[self.current]]
                test_labels = self.labels[self.labels[self.level]==self.range[self.current]]
            self.current += 1
            return train_labels, test_labels

class DataOrganizer:
    def __init__(self, name, exp_path, data_path, level):
        self.name = name
        self.exp_path = exp_path
        self.data_path = data_path
        # Specify the exact range of envs
        
        self.level = level
        assert level in {'env', 'subject', 'day'}
        print(f'Cross validation plan at {self.level} level')
        self.cross_validator = None
        
        # Data-Modality-Name
        self.data: dict = {}
        
        # Put all labels into one DataFrame
        self.total_segment_labels = pd.DataFrame(columns=['env', 'subject', 'bag', 'csi', 'group', 'segment', 'img', 'csi', 'img_inds', 'csi_inds'])
        
    def load(self):
        for dpath in self.data_path:
            paths = os.walk(dpath)
            tqdm.write(f'Loading {dpath}...')
            for path, _, file_lst in tqdm(paths, total=len(paths)):
                for file_name in file_lst:
                    file_name_, ext = os.path.splitext(file_name)
                    
                    # Load Label <subject>_matched.csv
                    if ext == '.csv' and 'matched' in file_name_:
                        sub_label = pd.read_csv(os.path.join(path, file_name))
                        self.total_segment_labels = pd.concat([self.total_segment_labels, sub_label], ignore_index=True)
                        
                    # Load CSI and IMGs <name>-<modality>.npy
                    elif ext == '.npy':
                        name, modality = file_name_.split('-')
                        if modality not in self.data.keys():
                            self.data[modality]: dict = {}

                        self.data[modality][name] = np.load(os.path.join(path, file_name), mmap_mode='r')
                                    
    def gen_loaders(self, transform=None, train_ratio=0.8, batch_size=64, save_dataset=False):
        if not self.cross_validator:
            self.cross_validator = CrossValidator(self.labels, self.level)

        train_labels, test_labels = next(self.cross_validator)
        
        if self.level == 'day':
            print('Different day validation:\n train = A308, test = A308T')
        else:
            print(f'Different {self.level} validation:\n'
                  f'train = {set(train_labels.loc[:, self.level].values)},\n'
                  f'test = {set(test_labels.loc[:, self.level].values)}\n'
                )

        # Store tags in dataset
        train_data = {key: [] for key in self.data.keys()}
        test_data = {key: [] for key in self.data.keys()}
        
        train_tags = train_labels.loc[:, ['env', 'subject', 'group', 'segment', 'img_id']]
        test_tags = test_labels.loc[:, ['env', 'subject', 'group', 'segment', 'img_id']]

        train_data['tag'] = train_tags.to_numpy()
        test_data['tag'] = test_tags.to_numpy()
        
        # Parse label into indices and regroup data
        for label, data in ((train_labels, train_data), (test_labels, test_data)):
            for env, subject, bag, csi, group, segment, img_id, csi_ids in label.itertuples():  
                for key in data.keys():
                    if key == 'csi':
                        data[key].append(self.data['csi'][csi][csi_ids])
                    elif key in ('rimg', 'cimg', 'bbx', 'center', 'depth'):
                        data[key].append(self.data[key][bag][img_id])
                        
        if save_dataset:
            pass

        train_set = MyDataset(train_data, transform)
        test_set = MyDataset(test_data, transform)
        
        # Generate loaders
        train_size = int(train_ratio * len(train_set))
        valid_size = len(train_set) - train_size
        train_set, valid_set = Data.random_split(train_set, [train_size, valid_size])
        train_loader = torch.utils.data.DataLoader(train_set, 
                                                   batch_size=self.batch_size, 
                                                   num_workers=14,
                                                   drop_last=True, 
                                                   pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_set, 
                                                   batch_size=self.batch_size, 
                                                   num_workers=14,
                                                   drop_last=True, 
                                                   pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_set, 
                                                   batch_size=1, 
                                                   num_workers=14,
                                                   pin_memory=True)
        
        print(f" exported train loader of len {len(train_loader)}, batch size = {self.batch_size}\n"
              f" exported valid loader of len {len(valid_loader)}, batch size = {self.batch_size}\n"
              f" exported test loader of len {len(test_loader)}, batch size = 1\n")
        
        return train_loader, valid_loader, test_loader