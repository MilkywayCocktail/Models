import torch
import torch.utils.data as Data
from torchvision import transforms
import numpy as np
import os
from PIL import Image


class MyDataset(Data.Dataset):
    """
    DATASET READER
    """
    def __init__(self, name, csi_path=None, img_path=None,
                 img_size=(128, 128), transform=None, int_image=False,
                 number=0, random=True,
                 mmap_mode='r'):
        """
        Wraps a dataset.\n
        :param name: the name of the dataset
        :param csi_path: path of x file (npy)
        :param img_path: path of y file (npy)
        :param img_size: original image size (height * width)
        :param transform: apply torchvision.transforms
        :param int_image: whether convert images to np.uint8. Default is False
        :param number: select a number of samples. Default is 0 (all)
        :param random: whether randomly choose images if number is specified. Default is True
        :param mmap_mode: mmap_mode='r' makes loading faster for large files
        """
        self.name = name
        self.paths = {'csi': csi_path, 'img': img_path}
        self.number = number
        self.random = random
        self.seeds = None
        self.img_size = img_size
        self.transform = transform
        self.int_img = int_image
        self.mmap_mode = mmap_mode
        self.data = {}

    def __convert__(self, sample):
        """
        Optionally convert a sample to np.uint8.
        :param sample: image (ndarray by default)
        :return: converted image
        """

        return np.squeeze(np.uint8(np.asarray(sample) * 255))

    def __transform__(self, sample):
        """
        Optionally apply transforms on images.\n
        :param sample: image (ndarray by default)
        :return: transformed image (tensor if transformed; ndarray if not transformed)
        """
        if self.transform:
            return self.transform(Image.fromarray(self.__convert__(sample)))
        else:
            return sample

    def __getitem__(self, index):
        """
        Retrieving samples.\n
        :param index: index of sample
        :return: csi, img, index
        """

        return {'csi': self.data['csi'][index],
                'img': self.__transform__(self.data['img'][index]),
                'ind': index}

    def __len__(self):
        return self.data['csi'].shape[0]

    def load_data(self):
        """
        Load data.\n
        :return: loaded dataset
        """
        print(f"{self.name} loading...")
        result = {}
        count = 0
        for key in self.paths.keys():
            if self.paths[key]:
                item = np.load(self.paths[key], mmap_mode=self.mmap_mode)
                result[key] = item
                count = item.shape[0]
                print(f"loaded {key} of {item.shape}")
            else:
                result[key] = None
                print(f"skipping {key}")

        if self.number != 0:
            if self.random:
                picked = np.random.choice(list(range(count)), size=self.number, replace=False)
            else:
                picked = np.arange(self.number)
            self.seeds = picked
            for key in self.paths.keys():
                result[key] = result[key][picked]

        self.data = result
        return result

# -------------------------------------------------------------------------- #
# MnistDataset
# Load MNIST from npy file
# Not updated
# -------------------------------------------------------------------------- #


class MnistDataset(MyDataset):
    """
    DATASET READER FOR MNIST
    """
    def __init__(self, name, mnist, img_size=(28, 28), transform=None, swap_xy=False, number=0):
        """
        Load MNIST data.
        :param mnist: path of mnist file (npy)
        :param img_size: original image size (height * width)
        :param transform: apply torchvision.transforms
        :param swap_xy: whether swap the x and y in dataset. Default is False
        :param number: select a number of samples. Default is 0 (all)
        """
        MyDataset.__init__(name=name, csi_path=None, img_path=None, img_size=img_size)
        self.swap_xy = swap_xy
        self.data = self.__load_data__(mnist, number=number)
        print('loaded')

    def __load_data__(self, mnist, number):
        """
        Load data.
        :param mnist: path of mnist file (npy)
        :param number: select a number of samples. Default is 0 (all)
        :return: loaded dataset
        """
        print(f"{self.name} loading...")
        x = mnist[:, 0].reshape((-1, 1, self.img_size[0], self.img_size[1]))
        y = mnist[:, 1]
        print(f"Loaded x {x.shape}, y {y.shape}")

        if number != 0:
            if x.shape[0] == y.shape[0]:
                total_count = x.shape[0]
                picked = np.random.choice(list(range(total_count)), size=number, replace=False)
                self.seeds = picked
                x = x[picked]
                y = y[picked]
        else:
            print("Lengths not equal!")

        if self.swap_xy:
            return {'x': y, 'y': x}
        else:
            return {'x': x, 'y': y}

# -------------------------------------------------------------------------- #
# DataSplitter
# Generate train, valid and test loaders
# Can choose to shuffle or not
# -------------------------------------------------------------------------- #


class DataSplitter:
    """
    DATASET SPLITTER (LOADER)
    """
    def __init__(self, data, train_ratio=0.8, valid_ratio=0.1, batch_size=64,
                 random=True, shuffle=True, generator=True):
        self.data = data
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.batch_size = batch_size
        self.random = random
        self.shuffle = shuffle
        self.generator = generator
        self.train_size, self.valid_size, self.test_size = self.__sizing__()

    def __sizing__(self):
        """
        Calculate train, valid, test sizes.
        :return: int sizes
        """
        train_size = int(len(self.data) * self.train_ratio)
        valid_size = int(len(self.data) * self.valid_ratio)
        test_size = int(len(self.data)) - train_size - valid_size
        return train_size, valid_size, test_size

    def unsplit_loader(self, batch_size=None, shuffle=None):
        """
        Export a loader without splitting.
        :param batch_size: default is 64
        :param shuffle: whether to shuffle samples. Default is True
        :return: data loader
        """
        if not batch_size:
            batch_size = self.batch_size
        if not shuffle:
            shuffle = self.shuffle
        print("Exporting...")
        loader = Data.DataLoader(self.data, batch_size=batch_size, shuffle=shuffle, drop_last=False)
        print(f"Exported loader of len {len(self.data)}")
        return loader

    def split_loader(self, test_batch_size=1, random=None, shuffle=None, generator=None,
                     num_workers=14, pin_memory=False):
        """
        Split the dataset into train, validation and test.
        :param test_batch_size: default is 1
        :param random: whether to split the dataset randomly. Default is True
        :param shuffle: whether to shuffle samples. Default is True
        :param generator: random seed generator for random split. Default is None
        :param num_workers: number of workers in DataLoader. Default is 14 (Server CPU is 32)
        :param pin_memory: whether to accelerate GPU reading. Default is False
        :return: train/valid/test dataloaders
        """
        if not random:
            random = self.random
        if not shuffle:
            shuffle = self.shuffle
        print("Exporting...")

        if random:
            train_dataset, valid_dataset, test_dataset = Data.random_split(
                self.data, [self.train_size, self.valid_size, self.test_size], generator=generator)
        else:
            r1 = self.train_size
            r2 = r1 + self.valid_size
            r3 = r2 + self.test_size
            train_dataset = torch.utils.data.Subset(self.data, range(r1))
            valid_dataset = torch.utils.data.Subset(self.data, range(r1, r2))
            test_dataset = torch.utils.data.Subset(self.data, range(r2, r3))

        train_loader = Data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=shuffle,
                                       num_workers=num_workers, drop_last=True, pin_memory=pin_memory)
        valid_loader = Data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=shuffle,
                                       num_workers=num_workers, drop_last=True, pin_memory=pin_memory)
        test_loader = Data.DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers, shuffle=shuffle,
                                      pin_memory=pin_memory)
        print(f"Exported loader len: train {len(train_loader)}, valid {len(valid_loader)}, test {len(test_loader)}")

        return train_loader, valid_loader, test_loader

# -------------------------------------------------------------------------- #
# MyDatasetV2
# Load csi, img (=c_img), bbx
# If the loss is "giou", you have to change bbx from xywh to xyxy.
# -------------------------------------------------------------------------- #


class MyDatasetV2(MyDataset):
    def __init__(self,
                 paths: dict,
                 bbx_ver='xywh',
                 *args,
                 **kwargs):
        super(MyDatasetV2, self).__init__(*args, **kwargs)

        self.bbx_ver = bbx_ver
        self.paths = paths
        self.modality = set()

    def adjust_bbx(self):
        if self.bbx_ver == 'xyxy':
            _bbx = np.zeros_like(self.data['bbx'])
            _bbx[..., 0:2] = self.data['bbx'][..., 0:2]
            _bbx[..., -1] = self.data['bbx'][..., -1] + self.data['bbx'][..., -3]
            _bbx[..., -2] = self.data['bbx'][..., -2] + self.data['bbx'][..., -4]
            self.data['bbx'] = _bbx

    def __getitem__(self, index):
        ret = {key: self.__transform__(value[index]) if key == 'img' else value[index]
               for key, value in self.data.items()
               }

        return ret

    def __len__(self):
        return self.data['csi'].shape[0]

    def load_data(self):
        """
        Load data.\n
        :return: loaded dataset
        """
        print(f"{self.name} loading...")
        result = {}
        count = 0
        for key, value in self.paths.items():
            if value:
                self.modality.add(key)
                item = np.load(value, mmap_mode=self.mmap_mode)
                result[key] = item
                count = item.shape[0]
                print(f"loaded {key} of {item.shape} as {item.dtype}")
            else:
                print(f"skipping {key}")

        if self.number != 0:
            if self.random:
                picked = np.random.choice(list(range(count)), size=self.number, replace=False)
            else:
                picked = np.arange(self.number)
            self.seeds = picked
            for key in self.paths.keys():
                result[key] = result[key][picked]

        self.data = result
        return result


class MyDatasetV3(MyDataset):
    def __init__(self,
                 paths: dict,
                 train_valid_id=None,
                 *args,
                 **kwargs):
        super(MyDatasetV3, self).__init__(*args, **kwargs)

        self.paths = paths
        self.modality = set()
        self.length = 0
        self.id = train_valid_id

    def __getitem__(self, index):
        ret = {key: self.__transform__(value[index]) if key == 'img' else value[index]
               for key, value in self.data.items()
               }

        return ret

    def __len__(self):
        return self.length

    def load_data(self):
        """
        Load data.\n
        :return: loaded dataset
        """
        print(f"{self.name} loading...")
        result = {'ind': None}
        for key, value in self.paths.items():
            if value:
                self.modality.add(key)
                item = np.load(value, mmap_mode=self.mmap_mode)
                if self.id is not None:
                    result[key] = item[self.id]
                else:
                    result[key] = item
                count = result[key].shape[0]
                print(f"loaded {key} of {result[key].shape} as {item.dtype}")
                self.length = count
            else:
                print(f"skipping {key}")

        if self.id is not None:
            result['ind'] = self.id
        else:
            result['ind'] = np.arange(self.length)
        self.data = result
        return result


class DataSplitterV2:
    def __init__(self, dataset: MyDatasetV3, batch_size=64, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def gen_loader(self, num_workers=14, pin_memory=False):
        print("Exporting loaders...")

        loader = Data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=num_workers, drop_last=True, pin_memory=pin_memory)
        print(f"{self.dataset.name} len {len(self.dataset)} - exported loader of len {len(loader)}")

        return loader


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
                          f"Data_path={self.data_path}\n"
                          f"gpu={self.gpu}\n"
                          )
        logfile.close()



