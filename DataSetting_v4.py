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
        
        self.range = ['A308T'] if self.level == 'day' else list(set(self.labels.loc[:, level].values))
        
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
                train_labels = self.labels[self.labels['env']!='A308T']
                test_labels = self.labels[self.labels['env']!='A308T']
            self.current += 1
            return train_labels, test_labels, self.range[self.current]

class DataOrganizer:
    def __init__(self, name, data_path, level):
        self.name = name
        self.data_path = data_path
        # Specify the exact range of envs
        
        self.level = level
        assert level in {'env', 'subject', 'day'}
        print(f'Cross validation plan at {self.level} level')
        
        # Data-Modality-Name
        self.data: dict = {}
        self.modalities = ['csi', 'rimg', 'cimg', 'bbx', 'ctr', 'dpt']
        
        # Put all labels into one DataFrame
        self.total_segment_labels = pd.DataFrame(columns=['env', 'subject', 'bag', 'csi', 'group', 'segment', 'img_inds', 'csi_inds'])
        self.train_labels = None
        self.test_labels = None
        self.current_test = None
        
    def load(self):
        for dpath in self.data_path:
            paths = os.walk(dpath)
            print(f'Loading {dpath}...\n')
            for path, _, file_lst in paths:
                for file_name in file_lst:
                    file_name_, ext = os.path.splitext(file_name)
                    
                    # Load Label <subject>_matched.csv
                    if ext == '.csv' and 'matched' in file_name_:
                        sub_label = pd.read_csv(os.path.join(path, file_name))
                        self.total_segment_labels = pd.concat([self.total_segment_labels, sub_label], ignore_index=True)
                        print(f'Loaded {file_name}')
                    # Load CSI and IMGs <name>-<modality>.npy
                    elif ext == '.npy':
                        if 'env' in file_name_ or 'test' in file_name_ or 'checkpoint' in file_name_:
                            continue
                        
                        name, modality = file_name_.split('-')
                        if modality in self.modalities:     
                            if modality not in self.data.keys():
                                self.data[modality]: dict = {}
                            self.data[modality][name] = np.load(os.path.join(path, file_name), mmap_mode='r')
                            print(f'Loaded {file_name}')
                     
        self.cross_validator = CrossValidator(self.total_segment_labels, self.level)   
          
    def gen_loaders(self, transform=None, train_ratio=0.8, batch_size=64, save_dataset=False):
        self.train_labels, self.test_labels, self.current_test = next(self.cross_validator)
        
        if self.level == 'day':
            print('Different day validation:\n train = A308, test = A308T')
        else:
            print(f'Different {self.level} validation:\n'
                  f'train = {set(self.train_labels.loc[:, self.level].values)},\n'
                  f'test = {self.current_test}\n'
                )

        # Store tags in dataset
        train_data = {key: None for key in self.data.keys()}
        test_data = {key: None for key in self.data.keys()}
        
        train_tags = self.train_labels.loc[:, ['env', 'subject', 'group', 'segment', 'img_inds']]
        test_tags = self.test_labels.loc[:, ['env', 'subject', 'group', 'segment', 'img_inds']]

        train_data['tag'] = train_tags.to_numpy()
        test_data['tag'] = test_tags.to_numpy()
        
        # Parse label into indices and regroup data
        tqdm.write('Assembling data...')
        for label, data in ((self.train_labels, train_data), (self.test_labels, test_data)):
            for i, env, subject, bag, csi, group, segment, img_inds, csi_inds, *_ in tqdm(label.itertuples(), total=len(label)):  
                csi_inds = eval(csi_inds.replace(' ', ','))
                for key in data.keys():
                    if key == 'csi':
                        # Add a new axis at the first dimension
                        # This new axis will disappear when __getitem__

                        ex_data = np.array(self.data['csi'][csi])[csi_inds]

                        data['csi'] = ex_data[np.newaxis, ...] if data['csi'] is None else np.concatenate((data['csi'], ex_data[np.newaxis, ...]))

                    elif key in ('rimg', 'cimg', 'bbx', 'center', 'depth'):
                        ex_data = self.data[key][bag][img_inds]
                        data[key] = ex_data if data[key] is None else np.concatenate((data[key], ex_data))
                        
        if save_dataset:
            pass

        print('Generating loaders...')
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
        
        return train_loader, valid_loader, test_loader, self.current_test
    