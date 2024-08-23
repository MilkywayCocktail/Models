import torch
import torch.utils.data as Data
import torch.distributed as dist
from torchvision import transforms
import numpy as np
import os
from PIL import Image
from tqdm.notebook import tqdm

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

        ret = {key: self.__transform__(value[index].copy()) if key in ('rimg', 'cimg') else torch.from_numpy(value[index].copy())
               for key, value in self.data.items()
               }

        return ret

    def __len__(self):
        return list(self.data.values())[0].shape[0]