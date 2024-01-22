import torch
import torch.utils.data as Data
import numpy as np
from PIL import Image


class MyDataset(Data.Dataset):
    """
    DATASET READER
    """
    def __init__(self, name, csi_path, img_path, img_size=(128, 128), transform=None, img='y', int_image=False, number=0,
                 mmap_mode='r'):
        """
        Wraps a dataset.\n
        :param name: the name of the dataset
        :param csi_path: path of x file (npy)
        :param img_path: path of y file (npy)
        :param img_size: original image size (height * width)
        :param transform: apply torchvision.transforms
        :param img: whether 'y' or 'x'. Default is 'y'
        :param int_image: whether convert images to np.uint8. Default is False
        :param number: select a number of samples. Default is 0 (all)
        :param mmap_mode: mmap_mode='r' makes loading faster for large files
        """
        self.name = name
        self.x_path = csi_path
        self.y_path = img_path
        self.number = number
        self.seeds = None
        self.img_size = img_size
        self.transform = transform
        self.img = img
        self.int_img = int_image
        self.mmap_mode = mmap_mode
        self.data = self.__load_data__()

    def __convert__(self, sample):
        """
        Optionally convert a sample to np.uint8.
        :param sample: image
        :return: converted image
        """
        if self.int_img:
            return np.uint8(np.array(sample * 255))
        else:
            return np.array(sample)

    def __transform__(self, sample):
        """
        Optionally apply transforms on images.\n
        :param sample: image
        :return: transformed image
        """
        if self.transform:
            return self.transform(Image.fromarray((self.__convert__(sample)).squeeze(), mode='L'))
        else:
            return self.__convert__(sample)

    def __getitem__(self, index):
        """
        Retrieving samples.\n
        :param index: index of sample
        :return: x, y, index
        """
        if self.img == 'y':
            return self.data['x'][index], self.__transform__(self.data['y'][index]), index
        elif self.img == 'x':
            return self.__transform__(self.data['x'][index]), self.data['y'][index], index

    def __len__(self):
        return self.data['x'].shape[0]

    def __load_data__(self):
        """
        Load data.\n
        :return: loaded dataset
        """
        print(f"{self.name} loading...")
        x = np.load(self.x_path, mmap_mode=self.mmap_mode)
        y = np.load(self.y_path, mmap_mode=self.mmap_mode)
        print(f"{self.name}: loaded x {x.shape}, y {y.shape}")
        if self.img == 'x':
            x = x.reshape((-1, 1, self.img_size[0], self.img_size[1]))
        elif self.img == 'y':
            y = y.reshape((-1, 1, self.img_size[0], self.img_size[1]))

        if self.number != 0:
            if x.shape[0] == y.shape[0]:
                total_count = x.shape[0]
                picked = np.random.choice(list(range(total_count)), size=self.number, replace=False)
                self.seeds = picked
                x = x[picked]
                y = y[picked]
            else:
                print("Lengths not equal!")

        return {'x': x, 'y': y}


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
        loader = Data.DataLoader(self.data, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        print(f"Exported loader of len {len(self.data)}...", end='')
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
        print(f"Exported loader len: train {len(train_dataset)}, valid {len(valid_dataset)}, test {len(test_dataset)}")

        return train_loader, valid_loader, test_loader


class MyDatasetBBX(MyDataset):
    def __init__(self, name,
                 csi_path, raw_img_path, crop_img_path, bbx_path,
                 img_size=(128, 128), transform=None, int_image=False,
                 bbx_ver='xywh',
                 number=0,
                 mmap_mode='r'):

        self.csi_path = csi_path
        self.raw_img_path = raw_img_path
        self.crop_img_path = crop_img_path
        self.bbx_path = bbx_path
        self.bbx_ver = bbx_ver
        super(MyDatasetBBX, self).__init__(name=name, csi_path=csi_path, img_path=None,
                                           img_size=img_size,
                                           transform=transform,
                                           int_image=int_image,
                                           number=number,
                                           mmap_mode=mmap_mode)

    def __getitem__(self, index):

        return self.data['csi'][index], \
               self.data['r_img'][index], \
               self.__transform__(self.data['c_img'][index]), \
               self.data['bbx'][index], \
               index

    def __len__(self):
        return self.data['csi'].shape[0]

    def __load_data__(self):
        print(f"{self.name} loading...")
        csi = np.load(self.csi_path, mmap_mode=self.mmap_mode)
        r_img = np.load(self.raw_img_path, mmap_mode=self.mmap_mode)
        c_img = np.load(self.crop_img_path, mmap_mode=self.mmap_mode)
        bbx = np.load(self.bbx_path)
        print(
            f"{self.name}: loaded csi {csi.shape}, raw_img {r_img.shape}, cropped_img {c_img.shape}, bbx {bbx.shape}")
        r_img = r_img.reshape((-1, 1, self.img_size[0], self.img_size[1]))
        c_img = c_img.reshape((-1, 1, 128, 128))
        # bbx is 'xywh' or 'xyxy'
        if self.bbx_ver == 'xyxy':
            bbx[..., -1] = bbx[..., -1] + bbx[..., -3]
            bbx[..., -2] = bbx[..., -2] + bbx[..., -4]

        if self.number != 0:
            if csi.shape[0] == r_img.shape[0]:
                total_count = csi.shape[0]
                picked = np.random.choice(list(range(total_count)), size=self.number, replace=False)
                self.seeds = picked
                csi = csi[picked]
                r_img = r_img[picked]
                c_img = c_img[picked]
                bbx = bbx[picked]

            else:
                print("Lengths not equal!")

        return {'csi': csi, 'r_img': r_img, 'c_img': c_img, 'bbx': bbx}


class MyDatasetBBX2(MyDataset):
    def __init__(self, name,
                 csi_path, img_path, bbx_path,
                 img_size=(128, 128), transform=None, int_image=False,
                 bbx_ver='xywh',
                 number=0,
                 mmap_mode='r'):

        self.csi_path = csi_path
        self.img_path = img_path
        self.bbx_path = bbx_path
        self.bbx_ver = bbx_ver
        super(MyDatasetBBX2, self).__init__(name=name, csi_path=csi_path, img_path=img_path,
                                            img_size=img_size,
                                            transform=transform,
                                            int_image=int_image,
                                            number=number,
                                            mmap_mode=mmap_mode)

    def __getitem__(self, index):

        return self.data['csi'][index], \
               self.__transform__(self.data['img'][index]), \
               self.data['bbx'][index], \
               index

    def __len__(self):
        return self.data['csi'].shape[0]

    def __load_data__(self):
        print(f"{self.name} loading...")
        csi = np.load(self.csi_path, mmap_mode=self.mmap_mode)
        img = np.load(self.img_path, mmap_mode=self.mmap_mode)
        bbx = np.load(self.bbx_path)
        print(f"{self.name}: loaded csi {csi.shape}, img {img.shape}, bbx {bbx.shape}")
        r_img = img.reshape((-1, 1, self.img_size[0], self.img_size[1]))
        # bbx is 'xywh' or 'xyxy'
        if self.bbx_ver == 'xyxy':
            bbx[..., -1] = bbx[..., -1] + bbx[..., -3]
            bbx[..., -2] = bbx[..., -2] + bbx[..., -4]

        if self.number != 0:
            if csi.shape[0] == r_img.shape[0]:
                total_count = csi.shape[0]
                picked = np.random.choice(list(range(total_count)), size=self.number, replace=False)
                self.seeds = picked
                csi = csi[picked]
                img = r_img[picked]
                bbx = bbx[picked]

            else:
                print("Lengths not equal!")

        return {'csi': csi, 'img': img, 'bbx': bbx}


class MyDatasetPDBBX2(MyDataset):
    def __init__(self, name,
                 pd_path, img_path, bbx_path,
                 img_size=(128, 128), transform=None, int_image=False,
                 bbx_ver='xywh',
                 number=0,
                 mmap_mode='r'):

        self.pd_path = pd_path
        self.img_path = img_path
        self.bbx_path = bbx_path
        self.bbx_ver = bbx_ver
        super(MyDatasetPDBBX2, self).__init__(name=name, csi_path=None, img_path=img_path,
                                            img_size=img_size,
                                            transform=transform,
                                            int_image=int_image,
                                            number=number,
                                            mmap_mode=mmap_mode)

    def __getitem__(self, index):

        return self.data['pd'][index], \
               self.__transform__(self.data['img'][index]), \
               self.data['bbx'][index], \
               index

    def __len__(self):
        return self.data['pd'].shape[0]

    def __load_data__(self):
        print(f"{self.name} loading...")
        pd = np.load(self.pd_path, mmap_mode=self.mmap_mode)
        img = np.load(self.img_path, mmap_mode=self.mmap_mode)
        bbx = np.load(self.bbx_path)
        print(f"{self.name}: loaded pd {pd.shape}, img {img.shape}, bbx {bbx.shape}")
        r_img = img.reshape((-1, 1, self.img_size[0], self.img_size[1]))
        # bbx is 'xywh' or 'xyxy'
        if self.bbx_ver == 'xyxy':
            bbx[..., -1] = bbx[..., -1] + bbx[..., -3]
            bbx[..., -2] = bbx[..., -2] + bbx[..., -4]

        if self.number != 0:
            if pd.shape[0] == r_img.shape[0]:
                total_count = pd.shape[0]
                picked = np.random.choice(list(range(total_count)), size=self.number, replace=False)
                self.seeds = picked
                pd = pd[picked]
                img = r_img[picked]
                bbx = bbx[picked]

            else:
                print("Lengths not equal!")

        return {'pd': pd, 'img': img, 'bbx': bbx}


class MyDatasetPDBBX3(MyDataset):
    def __init__(self, name,
                 csi_path, img_path, bbx_path, pd_path,
                 img_size=(128, 128), transform=None, int_image=False,
                 bbx_ver='xywh',
                 number=0,
                 mmap_mode='r'):

        self.pd_path = pd_path
        self.csi_path = csi_path
        self.img_path = img_path
        self.bbx_path = bbx_path
        self.bbx_ver = bbx_ver
        super(MyDatasetPDBBX3, self).__init__(name=name, csi_path=csi_path, img_path=img_path,
                                            img_size=img_size,
                                            transform=transform,
                                            int_image=int_image,
                                            number=number,
                                            mmap_mode=mmap_mode)

    def __getitem__(self, index):

        return self.data['csi'][index], \
               self.__transform__(self.data['img'][index]), \
               self.data['pd'][index], \
               self.data['bbx'][index], \
               index

    def __len__(self):
        return self.data['csi'].shape[0]

    def __load_data__(self):
        print(f"{self.name} loading...")
        csi = np.load(self.csi_path, mmap_mode=self.mmap_mode)
        img = np.load(self.img_path, mmap_mode=self.mmap_mode)
        pd = np.load(self.pd_path, mmap_mode=self.mmap_mode)
        bbx = np.load(self.bbx_path)
        print(f"{self.name}: loaded pd {pd.shape}, img {img.shape}, bbx {bbx.shape}")
        r_img = img.reshape((-1, 1, self.img_size[0], self.img_size[1]))
        # bbx is 'xywh' or 'xyxy'
        if self.bbx_ver == 'xyxy':
            bbx[..., -1] = bbx[..., -1] + bbx[..., -3]
            bbx[..., -2] = bbx[..., -2] + bbx[..., -4]

        if self.number != 0:
            if pd.shape[0] == r_img.shape[0]:
                total_count = pd.shape[0]
                picked = np.random.choice(list(range(total_count)), size=self.number, replace=False)
                self.seeds = picked
                pd = pd[picked]
                img = r_img[picked]
                bbx = bbx[picked]

            else:
                print("Lengths not equal!")

        return {'pd': pd, 'img': img, 'bbx': bbx}