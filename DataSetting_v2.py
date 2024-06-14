import torch
import torch.utils.data as Data
import torch.distributed as dist
from torchvision import transforms
import numpy as np
import os
from PIL import Image
from tqdm.notebook import tqdm

ver = 'V05'


class ExperimentInfo:
    def __init__(self, date, run, gpu, data_path):
        self.date = date
        self.run = run
        self.gpu = gpu
        self.data_path = data_path

    def log(self):
        save_path = f'../saved/{self.date}_{self.run}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(f"{save_path}ExpInfo.txt", 'w') as logfile:
            logfile.write(f"Experiment Info of {self.date}_{self.run}\n"
                          f"Data_dir={self.data_path}\n"
                          f"gpu={self.gpu}\n"
                          )
        logfile.close()


class DataRegrouper:
    version = ver

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data: dict = {}
        self.zero_segments = []
        self.modality = ['tag', 'depth', 'csi', 'center', 'pd', 'cimg', 'bbx', 'time', 'ind', 'rimg']

    def load_raw(self, modalities=None, scope=None, mmap_mode=None):
        # Filename: Txx_Gyy_Szz_mode.npy

        paths = os.walk(self.data_dir)
        tqdm.write('Loading dataset...')
        if scope:
            scope = {int(S.replace('T', '')) for S in scope}

        for path, dir_lst, file_lst in paths:
            for file_name in tqdm(file_lst):
                fname, ext = os.path.splitext(file_name)
                if ext == '.npy':
                    Take, Group, Segment, modality = fname.split('_')
                    Take = int(Take.replace('T', ''))
                    Group = int(Group.replace('G', ''))
                    Segment = int(Segment.replace('S', ''))

                    if scope and Take not in scope:
                        continue
                    if modalities and modality not in modalities:
                        continue
                    if [Take, Group, Segment] in self.zero_segments:
                        continue

                    din = np.load(os.path.join(path, file_name), mmap_mode=mmap_mode)
                    if len(din) > 0:
                        if Take not in self.data.keys():
                            self.data[Take]: dict = {}
                        if Group not in self.data[Take].keys():
                            self.data[Take][Group]: dict = {}
                        if Segment not in self.data[Take][Group].keys():
                            self.data[Take][Group][Segment]: dict = {}

                        self.data[Take][Group][Segment][modality] = din
                        self.data[Take][Group][Segment]['tag'] = np.array([[Take, Group, Segment]] * len(
                            self.data[Take][Group][Segment][modality]))
                        # tqdm.write(f'Loaded {fname} of len {len(self.data[Take][Group][Segment][modality])}')
                    else:
                        tqdm.write(f'Excluded {fname} len {len(din)}')
                        self.zero_segments.append([Take, Group, Segment])

    def regroup(self, takes):
        ret_data: dict = {}
        tqdm.write('Regrouping...')
        for modality in tqdm(self.modality):
            ret_data[modality] = []
            for Take in takes:
                _Take = int(Take.replace('T', ''))
                for Group in self.data[_Take].keys():
                    for Segment in self.data[_Take][Group].keys():
                        ret_data[modality].append(self.data[_Take][Group][Segment][modality])
                print(f"{Take} {modality} len={len(ret_data[modality])} ")
            try:
                ret_data[modality] = np.concatenate(ret_data[modality], axis=0)        
            except Exception as e:
                print(modality, e)
        # 'tag' = Take, Group, Segment, ind
        ret_data['tag'] = np.hstack((ret_data['tag'], ret_data['ind'].squeeze(axis=1))).astype(int)
            
        return ret_data
        

class ModalityLoader:
    def __init__(self, data_dir=None, modalities=None, mmap_mode=None, *args, **kwargs) -> None:
        self.data_dir = data_dir
        self.modalities = modalities
        self.data: dict = {}
        self.train_data: dict = {}
        self.test_data: dict = {}
        
        if data_dir:
            print(f"Loading from {data_dir}")
            paths = os.walk(self.data_dir)
            for path, _, file_lst in paths:
                for file_name in tqdm(file_lst):
                    file_name_, ext = os.path.splitext(file_name)
                    if ext == '.npy':
                        if 'train' in file_name_:
                            modality = file_name_.replace('_train', '')
                            if modalities and modality not in modalities:
                                continue
                            self.train_data[modality] = np.load(os.path.join(path, file_name), mmap_mode=mmap_mode)
                        elif 'test' in file_name_:
                            modality = file_name_.replace('_test', '')
                            if modalities and modality not in modalities:
                                continue
                            self.test_data[modality] = np.load(os.path.join(path, file_name), mmap_mode=mmap_mode)
                        else:
                            modality = file_name_
                            if modalities and modality not in modalities:
                                continue
                            self.data[modality] = np.load(os.path.join(path, file_name), mmap_mode=mmap_mode)
                        
        
    def profiling(self, scope):
        self.profile = np.zeros(list(self.data.values())[0].shape[0], dtype=bool)
        assert 'tag' in self.data.keys()
        ret_data: dict = {}
        for Take in scope:
            # Select specified takes
            _Take = int(Take.replace('T', ''))
            _take = np.where(self.data['tag'][:, 0] == _Take)
            self.profile[_take] = 1
            
        for mod, value in self.data.items():
            ret_data[mod] = value[self.profile]
        print(f'Profiled by {scope}')
        return ret_data
        

class MyDataset(Data.Dataset):
    """
    DATASET READER
    """
    version = ver

    def __init__(self, name,
                 data=None,
                 transform=None,
                 *args, **kwargs):

        self.name = name
        self.transform = transform
        self.data = data

    def __transform__(self, sample):
        """
        Optionally apply transforms on images.\n
        :param sample: image (ndarray by default)
        :return: transformed image (tensor if transformed; ndarray if not transformed)
        """
        if self.transform:
            return self.transform(Image.fromarray(np.squeeze(np.uint8(np.asarray(sample) * 255))))
        else:
            return sample

    def __getitem__(self, index):
        """
        Retrieving samples.\n
        :param index: index of sample
        :return: all modalities
        """

        ret = {key: self.__transform__(value[index]) if key in ('rimg', 'cimg') else torch.from_numpy(value[index])
               for key, value in self.data.items()
               }

        return ret

    def __len__(self):
        return list(self.data.values())[0].shape[0]

            

class DataSplitter:
    version = ver

    def __init__(self, dataset: MyDataset, batch_size=64, shuffle=True, distributed=False, gpu=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.distributed = distributed
        if self.distributed:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '5550'
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpu))
            dist.init_process_group(backend='nccl', init_method='env://', rank=torch.cuda.device_count(), world_size=1)

    def split_loader(self, train_ratio=0.8,  num_workers=14, pin_memory=True):
        print("Generating loaders...")
        train_size = int(train_ratio * len(self.dataset))
        valid_size = len(self.dataset) - train_size
        train_dataset, valid_dataset = Data.random_split(self.dataset, [train_size, valid_size])
        if not self.distributed:
            train_loader = Data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                           num_workers=num_workers, drop_last=True, pin_memory=pin_memory)
            valid_loader = Data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                           num_workers=num_workers, drop_last=True, pin_memory=pin_memory)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                       shuffle=self.shuffle, num_workers=num_workers,
                                                       drop_last=True, pin_memory=pin_memory, sampler=train_sampler)
            valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size,
                                                       shuffle=self.shuffle, num_workers=num_workers,
                                                       drop_last=True, pin_memory=pin_memory, sampler=valid_sampler)

        print(f" {self.dataset.name} len {len(self.dataset)}, distributed={self.distributed}\n"
              f" exported train loader of len {len(train_loader)}, batch size {self.batch_size}\n"
              f" exported valid loader of len {len(valid_loader)}, batch size {self.batch_size}\n")

        return train_loader, valid_loader

    def gen_loader(self, num_workers=14, pin_memory=False):
        print("Generating loaders...")
        if not self.distributed:
            loader = Data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                     num_workers=num_workers, drop_last=True, pin_memory=pin_memory)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
            loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                 shuffle=self.shuffle, num_workers=num_workers,
                                                 drop_last=True, pin_memory=pin_memory, sampler=sampler)
        print(f" {self.dataset.name} len {len(self.dataset)}, distributed={self.distributed}\n"
              f" exported loader of len {len(loader)}, batch size {self.batch_size}")

        return loader
