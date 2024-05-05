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
            self.track[key] = [kwargs[key]]

    def update(self):
        if self.updatable:
            for param, value in self.params.items():
                self.track[param].append(value.cpu().detach().tolist())

    def plot_track(self, *args: str):
        fig = plt.figure(constrained_layout=True)
        fig.suptitle("Extra Parameters")

        for param in args:
            if param in self.params.keys():
                plt.plot(self.track[param], label=param)
        plt.xlabel("#Epoch")
        plt.ylabel("Value")
        plt.grid()
        plt.legend(fontsize="20")
        plt.show()
        return fig, "Extra_Parameters.jpg"


class EarlyStopping:

    def __init__(self, save_path, patience=7, verbose=False, delta=0):

        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_valid_loss = -np.inf
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, *args, **kwargs):

        if val_loss > self.best_valid_loss:
            self.counter += 1
            print(f"\033[32mEarly Stopping reporting {self.counter} out of {self.patience}\033[0m")
            if self.counter >= self.patience:
                self.early_stop = True
            else:
                self.best_valid_loss = val_loss
                self.couner = 0


class BasicTrainer:
    def __init__(self, name, networks,
                 lr, epochs, cuda,
                 train_loader, valid_loader, test_loader,
                 save_path
                 ):
        self.name = name
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda:" + str(cuda) if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam

        self.extra_params = ExtraParams(self.device)
        self.models = {network.name: network.to(self.device) for network in networks
                       }

        self.dataloader = {'train': train_loader,
                           'valid': valid_loader,
                           'test': test_loader}

        self.modality = {'modality1', 'modality2', '...'}

        self.loss_terms = ('loss1', 'loss2', '...')
        self.pred_terms = ('predict1', 'predict2', '...')
        self.loss = MyLoss(self.name, self.loss_terms, self.pred_terms)
        self.temp_loss = {}
        self.best_val_loss = float("inf")

        self.save_path = save_path
        self.early_stopping = EarlyStopping()

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
            for idx, data in enumerate(self.dataloader['train'], 0):
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

                if idx % 5 == 0:
                    print(f"\r{self.name} train: epoch={epoch}/{self.epochs}, "
                          f"batch={idx}/{len(self.dataloader['train'])}, "
                          f"loss={self.temp_loss['LOSS'].item():.4f}, "
                          f"current best valid loss={self.best_val_loss:.4f}    ", end='', flush=True)

            for key, value in EPOCH_LOSS.items():
                EPOCH_LOSS[key] = np.average(value)
            self.loss.update('train', EPOCH_LOSS)
            self.extra_params.update()

            # =====================valid============================
            for model in train_module:
                self.models[model].eval()
            if eval_module:
                for model in eval_module:
                    self.models[model].eval()
            EPOCH_LOSS = {loss: [] for loss in self.loss_terms}

            for idx, data in enumerate(self.dataloader['valid'], 0):
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

                if idx % 5 == 0:
                    print(f"\r{self.name} valid: epoch={epoch}/{self.epochs}, "
                          f"batch={idx}/{len(self.dataloader['valid'])}, "
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
    def test(self, test_module=None, loader: str = 'test', **kwargs):
        if 'ind' not in self.modality:
            self.modality.add('ind')
        if not test_module:
            test_module = list(self.models.keys())
        for model in test_module:
            self.models[model].eval()

        EPOCH_LOSS = {loss: [] for loss in self.loss_terms}
        self.loss.reset('test', 'pred', dataset=loader)

        for idx, data in enumerate(self.dataloader[loader], 0):
            data_ = {}
            length = 0
            for key, value in data.items():
                if key in self.modality:
                    data_[key] = value.to(torch.float32).to(self.device)
                    length = len(value)

            with torch.no_grad():
                for sample in range(length):
                    data_i = {key: data_[key][sample][np.newaxis, ...] for key in data_.keys()}
                    PREDS = self.calculate_loss(data_i)

                    for key in EPOCH_LOSS.keys():
                        EPOCH_LOSS[key].append(self.temp_loss[key].item())

                    self.loss.update('pred', PREDS)

            if idx % 5 == 0:
                print(f"\r{self.name} test: sample={idx}/{len(self.dataloader[loader])}, "
                      f"loss={self.temp_loss['LOSS'].item():.4f}    ",
                      end='', flush=True)

        self.loss.update('test', EPOCH_LOSS)
        for key in EPOCH_LOSS.keys():
            EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])

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
                       f"{save_path}{notion}_{self.name}_{self.models[model]}@ep{self.current_ep()}.pth")
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
