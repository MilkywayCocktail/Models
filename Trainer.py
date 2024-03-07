import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from Loss import MyLoss
from misc import timer


class ExtraParams:
    def __init__(self, device):
        self.device = device
        self.params = {}
        self.track = {}
        self.updatable = False

    def add(self, **kwargs):
        if kwargs:
            self.updatable = True
        for key, value in kwargs.items():
            self.params[key] = torch.nn.Parameter(torch.tensor(value, device=self.device), requires_grad=True)
            self.track[key] = kwargs[key]

    def update(self):
        if self.updatable:
            for param, value in self.params.items():
                self.track[param].append(value.cpu().detach().tolist())

    def plot_track(self, *args: str):
        fig = plt.figure(constrained_layout=True)
        fig.suptitle("Extra Parameters")

        for param in args:
            if param in self.params.keys():
                plt.plot(self.params[param], label=param)

        plt.grid()
        plt.legend()
        plt.show()
        return fig, "Extra_Parameters.jpg"


class BasicTrainer:
    def __init__(self, name, networks,
                 lr, epochs, cuda,
                 train_loader, valid_loader, test_loader,
                 ):
        self.name = name
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda:" + str(cuda) if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam

        self.extra_params = ExtraParams(self.device)
        self.models = {network.name: network.to(self.device) for network in networks
                       }

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.modality = {'modality1', 'modality2', '...'}

        self.loss_terms = ('loss1', 'loss2', '...')
        self.pred_terms = ('predict1', 'predict2', '...')
        self.loss = MyLoss(self.name, self.loss_terms, self.pred_terms)
        self.temp_loss = {}
        self.best_val_loss = float("inf")

    def current_ep(self):
        return self.loss.epochs[-1]

    def calculate_loss(self, *inputs):
        # --- Return losses in this way ---
        self.temp_loss = {loss: 0 for loss in self.loss_terms}
        return {pred: None for pred in self.pred_terms}

    @timer
    def train(self, train_module=None, eval_module=None, autosave=False, notion='', **kwargs):
        if 'ind' not in self.modality:
            self.modality.add('ind')
        if not train_module:
            train_module = list(self.models.keys())
        params = [{'params': self.models[model].parameters()} for model in train_module]
        if self.extra_params.updatable:
            for param, value in self.extra_params.params.items():
                params.append({'params': value})
        optimizer = self.optimizer(params, lr=self.lr)

        # ===============train and validate each epoch==============
        for epoch in range(self.epochs):
            self.loss.logger(self.lr)
            # =====================train============================
            for model in train_module:
                self.models[model].train()
            if eval_module:
                for model in eval_module:
                    self.models[model].eval()
            EPOCH_LOSS = {loss: [] for loss in self.loss_terms}
            for idx, data in enumerate(self.train_loader, 0):
                data_ = {}
                for key, value in data.items():
                    if key in self.modality:
                        data_[key] = value.to(torch.float32).to(self.device)

                optimizer.zero_grad()
                PREDS = self.calculate_loss(data_)
                self.temp_loss['LOSS'].backward()
                optimizer.step()
                for key in EPOCH_LOSS.keys():
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

                if idx % (len(self.train_loader) // 5) == 0:
                    print(f"\r{self.name} train: epoch={epoch}/{self.epochs}, batch={idx}/{len(self.train_loader)}, "
                          f"loss={self.temp_loss['LOSS'].item():.4f}, "
                          f"current best valid loss={self.best_val_loss:.4f}    ", end='', flush=True)

            for key in EPOCH_LOSS.keys():
                EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
            self.loss.update('train', EPOCH_LOSS)
            self.extra_params.update()

            # =====================valid============================
            for model in train_module:
                self.models[model].eval()
            if eval_module:
                for model in eval_module:
                    self.models[model].eval()
            EPOCH_LOSS = {loss: [] for loss in self.loss_terms}

            for idx, data in enumerate(self.valid_loader, 0):
                data_ = {}
                for key, value in data.items():
                    if key in self.modality:
                        data_[key] = value.to(torch.float32).to(self.device)

                with torch.no_grad():
                    PREDS = self.calculate_loss(data_)
                for key in EPOCH_LOSS.keys():
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

                val_loss = np.average(EPOCH_LOSS['LOSS'])
                if 0 < val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

                if idx % (len(self.valid_loader) // 5) == 0:
                    print(f"\r{self.name} valid: epoch={epoch}/{self.epochs}, batch={idx}/{len(self.valid_loader)}, "
                          f"current best valid loss={self.best_val_loss:.4f}        ", end='', flush=True)

                if autosave:
                    save_path = f'../saved/{notion}/'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    with open(f"{save_path}{notion}_{self.name}.txt", 'w') as logfile:
                        logfile.write(f"{notion}_{self.name}\n"
                                      f"Best : val_loss={self.best_val_loss} @ epoch {self.current_ep()}\n"
                                      f"Modules:\n{list(self.models.values())}\n"
                                      )
                    logfile.close()

                    if 'save_model' in kwargs.keys() and kwargs['save_model'] is False:
                        pass
                    else:
                        for model in train_module:
                            torch.save(self.models[model].state_dict(),
                                       f"{save_path}{notion}_{self.models[model]}_best.pth")

            for key in EPOCH_LOSS.keys():
                EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
            self.loss.update('valid', EPOCH_LOSS)

    @timer
    def test(self, test_module=None, loader='test', **kwargs):
        if not test_module:
            test_module = list(self.models.keys())
        for model in test_module:
            self.models[model].eval()

        EPOCH_LOSS = {loss: [] for loss in self.loss_terms}

        if loader == 'test':
            loader = self.test_loader
        elif loader == 'train':
            loader = self.train_loader

        self.loss.reset('test', 'pred')

        for idx, data in enumerate(loader, 0):
            data_ = {}
            for key, value in data.items():
                if key in self.modality:
                    data_[key] = value.to(torch.float32).to(self.device)

            with torch.no_grad():
                for sample in range(loader.batch_size):
                    data_i = {key: data_[key][sample][np.newaxis, ...] for key in data_.keys()}
                    PREDS = self.calculate_loss(data_i)

                    for key in EPOCH_LOSS.keys():
                        EPOCH_LOSS[key].append(self.temp_loss[key].item())

                    self.loss.update('pred', PREDS)

            if idx % (len(loader)//5) == 0:
                print(f"\r{self.name} test: sample={idx}/{len(loader)}, loss={self.temp_loss['LOSS'].item():.4f}    ",
                      end='', flush=True)

        self.loss.update('test', EPOCH_LOSS)
        for key in EPOCH_LOSS.keys():
            EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
        for i in range(len(self.loss.loss['pred']['IND'])):
            self.loss.loss['pred']['IND'][i] = self.loss.loss['pred']['IND'][i].astype(int).tolist()

        print(f"\nTest finished. Average loss={EPOCH_LOSS}")

    def plot_train_loss(self, title=None, double_y=False, plot_terms='all', autosave=False, notion='', **kwargs):
        save_path = f'../saved/{notion}/'

        fig, filename = self.loss.plot_train(title, plot_terms, double_y)

        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            fig.savefig(f"{save_path}{notion}_{filename}")

    def plot_test(self, select_inds=None, select_num=8, autosave=False, notion='', **kwargs):
        # According to actual usages
        self.loss.generate_indices(select_inds, select_num)
        pass

    def save(self, notion=''):
        print("Saving models...")
        save_path = f'../saved/{notion}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for model in self.models:
            print(f"Saving {model}...")
            torch.save(self.models[model].state_dict(),
                       f"{save_path}{notion}_{self.models[model]}@ep{self.current_ep()}.pth")
        print("All saved!")

    def scheduler(self, turns=10,
                  lr_decay=False, decay_rate=0.4,
                  test_loader='train', select_num=8,
                  autosave=False, notion='', **kwargs):
        for i in range(turns):
            self.train(autosave=autosave, notion=notion, **kwargs)
            self.test(loader=test_loader, **kwargs)
            self.plot_train_loss(autosave=autosave, notion=notion, **kwargs)
            self.plot_test(select_num=select_num, autosave=autosave, notion=notion, **kwargs)
            if lr_decay:
                self.lr *= decay_rate

        print('\nSchedule Completed!')
