import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from Loss import MyLoss
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

        self.models = {network.name: network.to(self.device) for network in networks
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

    def current_ep(self):
        return self.loss.epochs[-1]

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

        # ===============train and validate each epoch==============
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
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                if autosave:
                    save_path = f'../saved/{notion}/'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                        logfile = open(f"{save_path}{notion}_{self.name}.txt", 'w')
                        logfile.write(f"{notion}_{self.name}\n"
                                      f"Modules: {list(self.models.values())}\n"
                                      f"Best : val_loss={best_val_loss} @ {self.current_ep()}")
                        for model in train_module:
                            torch.save(self.models[model].state_dict(),
                                       f"{save_path}{notion}_{self.models[model]}_best.pth")

            for key in EPOCH_LOSS.keys():
                EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
            self.loss.update('valid', EPOCH_LOSS)

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
        self.inds = None
        for key in EPOCH_LOSS.keys():
            EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
        print(f"\nTest finished. Average loss={EPOCH_LOSS}")

    def plot_train_loss(self, title=None, double_y=False, plot_terms='all', autosave=False, notion=''):
        if not title:
            title = f"{self.name} Training Status"
        filename = f"{notion}_{self.name}_train@{self.current_ep()}.jpg"

        save_path = f'../saved/{notion}/'

        fig = self.loss.plot_train(title, plot_terms, double_y)

        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig.savefig(f"{save_path}{filename}")
        plt.show()

    def plot_test(self):
        pass

    def generate_indices(self, source=None, select_num=8):
        if not source:
            source = self.loss.loss['pred']['IND']
        inds = np.random.choice(list(range(len(source))), select_num, replace=False)
        inds = np.sort(inds)
        self.inds = inds
        return inds

