import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torch.distributed as dist
from torchvision import transforms
import torch.nn.functional as F

import numpy as np
import pandas as pd
from scipy import signal
import os
from PIL import Image
import pickle

from tqdm.notebook import tqdm

remov = [('0709A10', 1, 8),
         ('0709A10', 1, 9),
         ('0709A53', 6, 6),
         ('0709A53', 6, 7)]


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
            
            # md = torch.zeros_like(pd, dtype=torch.cfloat)
            # for i in range(pd.shape[1]):
            #     real_filtered = torch.tensor(signal.medfilt(pd[:, i].real.cpu().numpy(), self.k), dtype=torch.float32, device=pd.device)
            #     imag_filtered = torch.tensor(signal.medfilt(pd[:, i].imag.cpu().numpy(), self.k), dtype=torch.float32, device=pd.device)
            #     print(real_filtered.shape, imag_filtered.shape)
            #     # md[:, i] = real_filtered + 1.j * imag_filtered
            # return torch.cat((real_filtered, imag_filtered), axis=-1).to(torch.float32)
            
            return torch.cat((torch.real(pd), torch.imag(pd)), axis=-1)
        
        try:
            # CSI shape = batch * 300 * 30 * 3
            # Reshape into batch * 3 * (30 * 300)
            u, *_ = torch.linalg.svd(csi.permute(0, 3, 2, 1).reshape(csi.shape[0], 3, -1), full_matrices=False)
            # AoA = batch * 4 (real & imag of 2)
            aoa = cal_pd(u)
            
            # Reshape into batch * 30 * (3 * 300)
            u, *_ = torch.linalg.svd(csi.permute(0, 2, 3, 1).reshape(csi.shape[0], 30, -1), full_matrices=False)
            # ToF = batch * 58 (real & imag of 29)
            tof = cal_pd(u)
            
            # Concatenate as a flattened vector
            pd = torch.cat((aoa, tof), axis=-1)

        except Exception as e:
            print(f'FilterPD aborted due to {e}')
        
        return pd
    
class MyDataset(Dataset):
    """
    DATASET wrapper
    Load CSI, IMG, IMG-related modalities (CIMG, DPT, CTR)
    """

    def __init__(self,
                 data,
                 label,
                 csi_len=300,
                 single_pd=True,
                 *args, **kwargs):

        self.data = data
        self.label = label
        self.alignment = 'tail'
        self.csi_len = csi_len
        self.single_pd = single_pd
        self.subject_code = {'higashinaka': 0,
                             'zhang': 1,
                             'chen': 2,
                             'wang': 3,
                             'jiao': 4,
                             'qiao': 5,
                             'zhang2': 6}
        self.env_code = {'A208': 0,
                         'A308': 1,
                         'B211': 2,
                         'C605': 3,
                         'A308T': 4}

    def __getitem__(self, index):
        """
        On-the-fly: select windowed CSI (and pd)
        """
        # Tag codes
        ret: dict = {}
        tag =  self.label.iloc[index][['env', 'subject', 'img_inds']]
        tag['env'] = self.env_code[tag['env']]
        tag['subject'] = self.subject_code[tag['subject']]
        ret['tag'] = tag.to_numpy().astype(int)
        
        # return the absolute index of sample
        ret['ind'] = self.label.index[index]

        bag = self.label.iloc[index]['bag']
        img_ind = int(self.label.iloc[index]['img_inds'])
        csi = self.label.iloc[index]['csi']
        csi_ind = int(self.label.iloc[index]['csi_inds'])
        pd_ind = int(self.label.iloc[index]['csi_inds'])


        for modality, value in self.data.items():
            if modality in ('rimg', 'cimg', 'bbx', 'ctr', 'dpt'):
                ret[modality] = np.copy(value[bag][img_ind])
                if modality == 'rimg':
                    ret[modality] = ret[modality][np.newaxis, ...] # Did not make spare axis

            elif modality == 'csi':

                if self.alignment == 'head':
                    csi_ind = np.arange(csi_ind, csi_ind + self.csi_len, dtype=int) 
                elif self.alignment == 'tail':
                    csi_ind = np.arange(csi_ind - self.csi_len, csi_ind, dtype=int)

                ret[modality] = np.copy(value[csi][csi_ind]) # Assume csi is n * 30 * 3
                
            elif modality == 'pd':
                if not self.single_pd:
                    if self.alignment == 'head':
                        pd_ind = np.arange(pd_ind, pd_ind + self.csi_len, dtype=int) 
                    elif self.alignment == 'tail':
                        pd_ind = np.arange(pd_ind - self.csi_len, pd_ind, dtype=int)
                        
                ret[modality] = np.copy(value[csi][pd_ind])
                
        return ret

    def __len__(self):
        return len(self.label)
    
class Preprocess:
    def __init__(self, new_size=(128, 128), filter_pd=False):
        self.new_size = new_size
        self._filterpd = FilterPD()
        self.filter_pd = filter_pd

    def transform(self, tensor):
        return F.interpolate(tensor, size=self.new_size, mode='bilinear', align_corners=False)
    
    def __call__(self, data, modalities):
        """
        Manually perform lazy preprocess
        """
        
        # Adjust key name
        data['center'] = data['ctr']
        data['depth'] = data['dpt']
        
        #  Transform images
        if self.new_size and 'rimg' in modalities:
            data['rimg'] = self.transform(data['rimg'])

        if 'csi' in modalities:
            if self.filter_pd:
                # Filter and reshape CSI, calculate phasediff
                # batch * packet * sub * rx
                csi_real = torch.tensor(signal.savgol_filter(np.real(data['csi']), 21, 3, axis=1), 
                                        dtype=torch.float32, device=data['csi'].device)  # denoise for real part
                csi_imag = torch.tensor(signal.savgol_filter(np.imag(data['csi']), 21, 3, axis=1), 
                                        dtype=torch.float32, device=data['csi'].device) # denoise for imag part
                csi_complex = csi_real + 1.j * csi_imag
                
                # batch *  packet * sub * (rx * 2)
                csi = torch.cat((csi_real, csi_imag), axis=-1)
                
                # batch *  (rx * 2) * sub * packet
                csi = csi.permute(0, 3, 2, 1)

                data['csi'] = csi
                data['pd'] = self._filterpd(csi_complex)
            
            else:
                csi = torch.cat((torch.real(data['csi']), torch.imag(data['csi'])), axis=-1)
                csi = csi.permute(0, 3, 2, 1)
                data['csi'] = csi
        
        return data

class CrossValidator:
    """
    Generate labels for cross validation
    """
    def __init__(self, labels, level, subset_ratio=1):
        self.labels = labels
        self.level = level
        self.subset_ratio = subset_ratio
        
        self.range = ['A308', 'A308T'] if self.level == 'day' else list(set(self.labels.loc[:, level].values))
        self.current = -1
        self.current_test = None

    def __iter__(self):
        return self
    
    def __next__(self):
        self.current += 1
        
        if self.current >= len(self.range):
            raise StopIteration
        
        self.current_test = self.range[self.current]
        print(f"Fetched level {self.level}, {self.current + 1} of {len(self.range)}, current = {self.current_test}")
                            
        ran = set(self.range)
        ran.remove(self.current_test)

        if self.level == 'day':
            train_labels = self.labels[self.labels['env']!=self.current_test]
            test_labels = self.labels[self.labels['env']==self.current_test]

        else:
            # Select one as leave-1-out test
            train_labels = self.labels[self.labels[self.level]!=self.current_test]
            test_labels = self.labels[self.labels[self.level]==self.current_test]
                 

        if self.subset_ratio < 1:
            
            train_subset_size = int(len(train_labels) * self.subset_ratio)
            test_subset_size = int(len(test_labels) * self.subset_ratio) 
            
            print(f" Train set range = {ran}, len = {train_subset_size} from {len(train_labels)}\n"
                  f" Test set current = {self.current_test}, len = {test_subset_size} from {len(test_labels)}"
                  )

            train_subset_indices = torch.randperm(len(train_labels))[:train_subset_size]
            test_subset_indices = torch.randperm(len(test_labels))[:test_subset_size]

            train_labels = train_labels.iloc[train_subset_indices]
            test_labels = test_labels.iloc[test_subset_indices]
            
        else:
            print(f" Train set range = {ran}, len = {len(train_labels)}\n"
                  f" Test set current = {self.current_test}, len = {len(test_labels)}")

        return (train_labels, test_labels, self.current_test)
    

class Removal:
    """
    Remove unpaired segments from dataset.
    """
    def __init__(self, conditions):
        self.conditions = conditions
        # (csi, group, segment) tuples
        
    def __call__(self, labels):
        for (csi, group, segment) in self.conditions:
            removal = (labels['csi']==csi) & (labels['group']==group) & (labels['segment']==segment)
            labels = labels.loc[~removal]
        return labels


class DataOrganizer:
    def __init__(self, name, data_path, level):
        self.name = name
        self.data_path = data_path
        # Specify the exact range of envs
        
        self.level = level
        assert level in {'env', 'subject', 'day'}
        print(f'Cross validation plan at {self.level} level')
        
        self.batch_size = 64
        
        # Data-Modality-Name
        self.data: dict = {}
        self.modalities = ['csi', 'rimg', 'cimg', 'bbx', 'ctr', 'dpt', 'pd']
        
        # Put all labels into one DataFrame
        self.total_segment_labels = pd.DataFrame(columns=['env', 'subject', 'bag', 'csi', 'group', 'segment', 'timestamp', 'img_inds', 'csi_inds'])
        self.train_indicies = None
        self.test_indicies = None        
        self.train_labels = None
        self.test_labels = None
        self.current_test = None
        
        self.cross_validator = None
        
        self.removal = Removal(remov)
        
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
                        print(f'Loaded {file_name} of len {len(sub_label)}')
                        
                    # Load CSI and IMGs <name>-<modality>.npy
                    elif ext == '.npy':
                        if 'env' in file_name_ or 'test' in file_name_ or 'checkpoint' in file_name_:
                            continue
                        
                        name, modality = file_name_.split('-')
                        if modality in self.modalities:     
                            if modality not in self.data.keys():
                                self.data[modality]: dict = {}
                            self.data[modality][name] = np.load(os.path.join(path, file_name), mmap_mode='r')
                            print(f'Loaded {file_name} of shape {self.data[modality][name].shape}')
                     
        # self.total_segment_labels['csi_inds'] = self.total_segment_labels['csi_inds'].apply(lambda x: list(map(int, x.strip('[]').split())))
            
    def gen_plan(self, subset_ratio=1, save=False, notion=''):
        if not self.cross_validator:
            self.cross_validator = CrossValidator(self.total_segment_labels, self.level, subset_ratio)   
        
        if save:
            print(f'Saving plan {self.level} @ {subset_ratio}...')
            cross_validator = CrossValidator(self.total_segment_labels, self.level, subset_ratio) 
            with open(f'../dataset/Door_EXP/{self.level}_r{subset_ratio}_{notion}.pkl', 'wb') as f:
                plan = list(cross_validator)
                pickle.dump(plan, f)
                
            print('Plan saved!\n')
            
        # Divide train and test
        self.train_labels, self.test_labels, self.current_test = next(self.cross_validator)
    
    def load_plan(self, path):
        with open(path, 'rb') as f:
            plan = pickle.load(f)
        self.cross_validator = iter(plan)
        print(f'Loaded plan!')
    
    def gen_loaders(self, mode='s', train_ratio=0.8, batch_size=64, csi_len=300, single_pd=True, num_workers=14, save_dataset=False):

        print(f'Generating loaders for {mode}: level = {self.level}, current test = {self.current_test}')
        data = self.data.copy()
        
        if mode == 't':
            data.pop('csi')
            data.pop('pd')

        else:
            # if mode == 's'
            self.train_labels = self.removal(self.train_labels)
            self.test_labels = self.removal(self.test_labels)
            
            if mode == 'c':
                data = self.data.copy()
                data.pop('pd')
                data.pop('cimg')
                
        dataset = MyDataset(data, self.train_labels, csi_len, single_pd)
        test_dataset = MyDataset(data, self.test_labels, csi_len, single_pd)
            
        print(f' Train dataset length = {len(dataset)}\n'
              f' Test dataset length = {len(test_dataset)}')
        
        # Generate loaders
        train_size = int(train_ratio * len(dataset))
        valid_size = len(dataset) - train_size
        train_set, valid_set = random_split(dataset, [train_size, valid_size])
        train_loader = DataLoader(train_set, 
                                  batch_size=batch_size, 
                                  num_workers=num_workers,
                                  drop_last=True, 
                                  pin_memory=True)
        valid_loader = DataLoader(valid_set, 
                                    batch_size=batch_size, 
                                    num_workers=num_workers,
                                    drop_last=True, 
                                    pin_memory=True)
        test_loader = DataLoader(test_dataset, 
                                    batch_size=batch_size, 
                                    num_workers=num_workers,
                                    pin_memory=True)
        
        print(f" Exported train loader of len {len(train_loader)}, batch size = {batch_size}\n"
              f" Exported valid loader of len {len(valid_loader)}, batch size = {batch_size}\n"
              f" Exported test loader of len {len(test_loader)}, batch size = {batch_size}\n")
        
        return train_loader, valid_loader, test_loader, self.current_test
    