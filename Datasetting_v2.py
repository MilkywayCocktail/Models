import torch
import torch.utils.data as Data
from torchvision import transforms
import numpy as np
import os
from PIL import Image
from tqdm.notebook import tqdm

ver = 'V05'


class DataPlanner:
    version = ver

    def __init__(self, data_dir, mode='valid'):
        self.data_dir = data_dir
        self.mode = mode
        self.data: dict = {}
        self.modality = []

    def load_raw(self):
        # Filename: Txx_Gyy_Szz_mode.npy

        paths = os.walk(self.data_dir)
        tqdm.write('Loading dataset...')

        for path, dir_lst, file_lst in paths:
            for file_name in tqdm(file_lst):
                fname, ext = os.path.splitext(file_name)
                if ext == '.npy':
                    Take, Group, Segment, modality = fname.split('_')[-1]
                    if modality not in self.modality:
                        self.modality.extend(modality)

                    if Take not in self.data.keys():
                        self.data[Take]: dict = {}
                    if Group not in self.data[Take].keys():
                        self.data[Take][Group]: dict = {}
                    if Segment not in self.data[Take][Group].keys():
                        self.data[Take][Group][Segment]: dict = {}

                    self.data[Take][Group][Segment][modality] = np.load(os.path.join(path, file_name))
                    self.data[Take][Group][Segment]['fname'] = fname

    def regroup(self, takes):
        ret_data: dict = {}
        tqdm.write('Regrouping...')
        for modality in tqdm(self.modality):
            ret_data[modality] = []
            for Take in takes:
                for Group in self.data[Take].keys():
                    for Segment in self.data[Take][Group].keys():
                        ret_data[modality].append(self.data[Take][Group][Segment][modality])

            ret_data[modality] = np.concatenate(ret_data[modality], axis=-1)
        return ret_data


class MyDataset(Data.Dataset):
    """
    DATASET READER
    """
    version = ver

    def __init__(self, name,
                 data,
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
            return self.transform(Image.fromarray(sample))
        else:
            return sample

    def __getitem__(self, index):
        """
        Retrieving samples.\n
        :param index: index of sample
        :return: csi, img, index
        """

        ret = {key: self.__transform__(value[index]) if key == 'img' else value[index]
               for key, value in self.data.items() if key != 'fname'
               }
        ret['ind'] = f"{self.data['fname'][index]}_{self.data['ind'][index]}"

        return ret

    def __len__(self):
        return self.data['ind'].shape[0]


class DataSplitter:
    version = ver

    def __init__(self, dataset: MyDataset, batch_size=64, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def split_loader(self, train_ratio=0.8):
        train_size = int(train_ratio * len(self.dataset))
        valid_size = len(self.dataset) - train_size
        train_dataset, valid_dataset = Data.random_split(self.dataset, [train_size, valid_size])
        return train_dataset, valid_dataset

    def gen_loader(self, num_workers=14, pin_memory=False):
        print("Exporting loaders...")

        loader = Data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                 num_workers=num_workers, drop_last=True, pin_memory=pin_memory)
        print(f"{self.dataset.name} len {len(self.dataset)} - "
              f"exported loader of len {len(loader)}, batch size {self.batch_size}")

        return loader
