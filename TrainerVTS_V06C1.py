import torch
import torch.nn as nn
from torchvision.ops import generalized_box_iou_loss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.ops import generalized_box_iou_loss
import os
from Loss import MyLoss, MyLossBBX
from misc import timer


class BasicTrainer:
    def __init__(self, name, networks,
                 lr, epochs, cuda,
                 train_loader, valid_loader, test_loader):
        self.name = name
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda:" + str(cuda) if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam

        self.models = {network.name: network for network in networks
                       }

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.using_datatype = ()

        self.loss_terms = []
        self.pred_terms = []
        self.loss = MyLoss()
        self.temp_loss = {}
        self.inds = None

    def current_title(self):
        return f"{self.name}@{self.loss.epochs[-1]}"

    def calculate_loss(self, *inputs):
        self.temp_loss = {loss: 0 for loss in self.loss_terms}
        return {pred: None for pred in self.pred_terms}

    @timer
    def train(self, train_module=None, eval_module=None, autosave=False, notion=''):
        optimizer = self.optimizer([{'params': self.models[model].parameters()} for model in train_module], lr=self.lr)
        self.loss.logger(self.lr, self.epochs)
        best_val_loss = float("inf")
        if not train_module:
            train_module = list(self.models.keys())

        for epoch in range(self.epochs):
            # =====================train============================
            for model in train_module:
                self.models[model].train()
            if eval_module:
                for model in eval_module:
                    self.models[model].eval()
            EPOCH_LOSS = {loss: [] for loss in self.loss_terms}
            for idx, data in enumerate(self.train_loader, 0):
                for key in data.keys():
                    if key in self.using_datatype:
                        data[key] = data[key].to(torch.float32).to(self.device)
                    else:
                        data.pop(key)

                optimizer.zero_grad()
                PREDS = self.calculate_loss(data)
                self.temp_loss['LOSS'].backward()
                optimizer.step()
                for key in EPOCH_LOSS.keys():
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

                if idx % (len(self.train_loader) // 5) == 0:
                    print(f"\r{self.name}: epoch={epoch}/{self.epochs}, batch={idx}/{len(self.train_loader)}, "
                          f"loss={self.temp_loss['LOSS'].item():.4f}", end='')

            for key in EPOCH_LOSS.keys():
                EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
            self.loss.update('train', EPOCH_LOSS)

            # =====================valid============================
            for model in train_module:
                self.models[model].eval()
            if eval_module:
                for model in eval_module:
                    self.models[model].eval()
            EPOCH_LOSS = {loss: [] for loss in self.loss_terms}

            for idx, data in enumerate(self.train_loader, 0):
                for key in data.keys():
                    if key in self.using_datatype:
                        data[key] = data[key].to(torch.float32).to(self.device)
                    else:
                        data.pop(key)
                with torch.no_grad():
                    PREDS = self.calculate_loss(data)
                for key in EPOCH_LOSS.keys():
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

                val_loss = np.average(EPOCH_LOSS['LOSS'])

                if autosave:
                    save_path = f'../saved/{notion}/'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        logfile = open(f"{save_path}{notion}_best_t.txt", 'w')
                        logfile.write(f"{self.name} best : {self.current_title()}")
                        for model in train_module:
                            torch.save(self.models[model].state_dict(),
                                       f"{save_path}{notion}_{self.models[model]}_best.pth")

            for key in EPOCH_LOSS.keys():
                EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
            self.loss['t'].update('valid', EPOCH_LOSS)

    @timer
    def test(self, test_module=None, eval_module=None, loader='test'):
        if not test_module:
            test_module = list(self.models.keys())
        for model in test_module:
            self.models[model].eval()
        if eval_module:
            for model in eval_module:
                self.models[model].eval()
        EPOCH_LOSS = {loss: [] for loss in self.loss_terms}

        if loader == 'test':
            loader = self.test_loader
        elif loader == 'train':
            loader = self.train_loader

        self.loss.reset('test')
        self.loss.reset('pred')

        for idx, data in enumerate(loader, 0):
            for key in data.keys():
                if key in self.using_datatype:
                    data[key] = data[key].to(torch.float32).to(self.device)
                else:
                    data.pop(key)

            with torch.no_grad():
                for sample in range(loader.batch_size):
                    _data = {key: data[key][sample][np.newaxis, ...] for key in data.keys()}
                    PREDS = self.calculate_loss(_data)

                    for key in EPOCH_LOSS.keys():
                        EPOCH_LOSS[key].append(self.temp_loss[key].item())

                    self.loss.update('pred', PREDS)

            if idx % (len(loader)//5) == 0:
                print(f"\r{self.name}: test={idx}/{len(loader)}, loss={self.temp_loss['LOSS'].item():.4f}", end='')

        self.loss.update('test', EPOCH_LOSS)
        for key in EPOCH_LOSS.keys():
            EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
        print(f"\nTest finished. Average loss={EPOCH_LOSS}")


class TeacherTrainer:
    def __init__(self, networks,
                 lr, epochs, cuda,
                 train_loader, valid_loader, test_loader):
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda:" + str(cuda) if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam

        self.models = {network.module: network for network in networks
                       }

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.alpha = 0.8
        self.beta = 1.2

        self.recon_lossfunc = nn.MSELoss(reduction='sum')

        self.loss = {'t': MyLoss(loss_terms=['LOSS', 'KL', 'RECON'],
                                 pred_terms=['GT', 'PRED', 'LAT', 'IND'])}

        self.temp_loss = {}
        self.inds = None

    def vae_loss(self, pred, gt, mu, logvar):
        recon_loss = self.recon_lossfunc(pred, gt) / pred.shape[0]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss * self.beta
        return loss, kl_loss, recon_loss

    def calculate_loss_t(self, img, i=None):

        z, mu, logvar = self.models['imgen'](img)
        output = self.models['imgde'](z)
        loss, kl_loss, recon_loss = self.vae_loss(output, img, mu, logvar)

        self.temp_loss = {'LOSS': loss,
                          'KL': kl_loss,
                          'RECON': recon_loss
                          }
        return {'GT': img,
                'PRED': output,
                'LAT': torch.cat((mu, logvar), -1),
                'IND': i
                }

    @timer
    def train_teacher(self, autosave=False, notion=''):
        """
        Trains the teacher.
        :param autosave: whether to save model parameters. Default is False
        :param notion: additional notes in save name
        :return: trained teacher
        """

        optimizer = self.optimizer([{'params': self.models['imgen'].parameters()},
                                    {'params': self.models['imgde'].parameters()}], lr=self.lr)
        self.loss['t'].logger(self.lr, self.epochs)
        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            # =====================train============================
            self.models['imgen'].train()
            self.models['imgde'].train()
            EPOCH_LOSS = {'LOSS': [],
                          'KL': [],
                          'RECON': []
                          }
            for idx, (csi, img, pd, bbx, index) in enumerate(self.train_loader, 0):
                img = img.to(torch.float32).to(self.device)
                optimizer.zero_grad()

                PREDS = self.calculate_loss_t(img)
                self.temp_loss['LOSS'].backward()
                optimizer.step()
                for key in EPOCH_LOSS.keys():
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

                if idx % (len(self.train_loader) // 5) == 0:
                    print(f"\rTeacher: epoch={epoch}/{self.epochs}, batch={idx}/{len(self.train_loader)}, "
                          f"loss={self.temp_loss['LOSS'].item():.4f}", end='')

            for key in EPOCH_LOSS.keys():
                EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
            self.loss['t'].update('train', EPOCH_LOSS)

            # =====================valid============================
            self.models['imgen'].eval()
            self.models['imgde'].eval()

            EPOCH_LOSS = {'LOSS': [],
                          'KL': [],
                          'RECON': []
                          }

            for idx, (csi, img, pd, bbx, index) in enumerate(self.valid_loader, 0):
                img = img.to(torch.float32).to(self.device)
                with torch.no_grad():
                    PREDS = self.calculate_loss_t(img)
                for key in EPOCH_LOSS.keys():
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

                val_loss = np.average(EPOCH_LOSS['LOSS'])

                if autosave:
                    save_path = f'../saved/{notion}/'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        logfile = open(f"{save_path}{notion}_best_t.txt", 'w')
                        logfile.write(f"Teacher best : {self.current_title()}")
                        torch.save(self.models['imgen'].state_dict(),
                                   f"{save_path}{notion}_{self.models['imgen']}_best.pth")
                        torch.save(self.models['imgde'].state_dict(),
                                   f"{save_path}{notion}_{self.models['imgde']}_best.pth")

            for key in EPOCH_LOSS.keys():
                EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
            self.loss['t'].update('valid', EPOCH_LOSS)

    def test_teacher(self, loader='test'):
        self.models['imgen'].eval()
        self.models['imgde'].eval()

        EPOCH_LOSS = {'LOSS': [],
                      'KL': [],
                      'RECON': []
                      }

        if loader == 'test':
            loader = self.test_loader
        elif loader == 'train':
            loader = self.train_loader

        self.loss['t'].reset('test')
        self.loss['t'].reset('pred')

        for idx, (csi, img, pd, bbx, index) in enumerate(loader, 0):
            img = img.to(torch.float32).to(self.device)
            with torch.no_grad():
                for sample in range(loader.batch_size):
                    ind_ = index[sample][np.newaxis, ...]
                    img_ = img[sample][np.newaxis, ...]
                    PREDS = self.calculate_loss_t(img_, ind_)

                    for key in EPOCH_LOSS.keys():
                        EPOCH_LOSS[key].append(self.temp_loss[key].item())

                    self.loss['t'].update('pred', PREDS)

            if idx % (len(loader)//5) == 0:
                print(f"\rTeacher: test={idx}/{len(loader)}, loss={self.temp_loss['LOSS'].item():.4f}", end='')

        self.loss['t'].update('test', EPOCH_LOSS)
        for key in EPOCH_LOSS.keys():
            EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
        print(f"\nTest finished. Average loss={EPOCH_LOSS}")