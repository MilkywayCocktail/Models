import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import os
import matplotlib.pyplot as plt
from Loss import MyLossLog
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

    def __init__(self, min_epoch=200, early_stop_max=20, lr_decay_max=5, verbose=True, delta=0, *args, **kwargs):

        self.min_epoch = min_epoch

        self.early_stop_max = early_stop_max
        self.early_stop_counter = 0
        self.stop_flag = False

        self.verbose = verbose
        self.delta = delta
        self.total_epochs = 0
        self.best_valid_loss = np.inf

        self.decay_flag = False
        self.lr_decay_counter = 0
        self.lr_decay_max = lr_decay_max

    def __call__(self, val_loss, early_stop=True, lr_decay=True):
        self.total_epochs += 1
        self.decay_flag = False
        if self.min_epoch !=0 and self.total_epochs < self.min_epoch:
            return
        
        if early_stop:
            if val_loss >= self.best_valid_loss:
                self.early_stop_counter += 1
                if self.verbose:
                    print(f"\033[32mEarly Stopping reporting: {self.early_stop_counter} out of {self.early_stop_max}\033[0m")
                if self.early_stop_counter >= self.early_stop_max:
                    if lr_decay:
                        self.lr_decay_counter += 1
                        if self.verbose:
                            print(f"\033[32mLr decay reporting: {self.lr_decay_counter} out of {self.lr_decay_max}. "
                                f"Decay rate = {0.5 ** self.lr_decay_counter}\033[0m")
                        if self.lr_decay_counter >= self.lr_decay_max:
                            self.stop_flag = True
                        else:
                            self.decay_flag = True
                            self.early_stop_counter = 0
                    else:
                        self.stop_flag = True
            else:
                self.best_valid_loss = val_loss
                self.early_stop_counter = 0


class BasicTrainer:
    def __init__(self, name, 
                 epochs, cuda,
                 train_loader, valid_loader, test_loader,
                 loss_optimizer: dict,
                 networks = None,
                 notion = None,
                 *args, **kwargs
                 ):
        self.name = name
        self.epochs = epochs
        self.loss_optimizer = loss_optimizer

        # Please define optimizers in this way
        # self.optimizer: dict = {'LOSS1': ['optimizer1', lr1],
        #                        'LOSS2': ['optimizer2', lr2],
        #                        'LOSS3': '...'}
        self.thread = 'single'

        self.dataloader = {'train': train_loader,
                           'valid': valid_loader,
                           'test': test_loader}

        if isinstance(cuda, int):
            self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
            self.extra_params = ExtraParams(self.device)
            if networks:
                self.models = {network.name: network.to(self.device) for network in networks
                            }
        elif isinstance(cuda, list) or isinstance(cuda, tuple) or isinstance(cuda, set):
            self.thread = 'multi'
            self.ddp_setup(cuda=cuda)
            self.extra_params = ExtraParams(self.device)
            if networks:
                self.models = {network.name: DDP(network.cuda()) for network in networks
                            }

        self.modality = {'modality1', 'modality2', '...'}

        self.loss_terms = ('loss1', 'loss2', '...')
        self.pred_terms = ('predict1', 'predict2', '...')
        self.losslog = MyLossLog(self.name, self.loss_terms, self.pred_terms)
        self.temp_loss = {}
        self.best_val_loss = float("inf")
        self.best_vloss_ep = 0

        self.notion = notion
        self.save_path = f'../saved/{notion}/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.early_stopping = None

    @staticmethod
    def ddp_setup(cuda):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, cuda))
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '5800'
        dist.init_process_group(backend='nccl', init_method='env://', rank=torch.cuda.device_count(), world_size=1)

    def data_to_device(self, data):
        if self.thread == 'single':
            return data.to(torch.float32).to(self.device)
        elif self.thread == 'multi':
            return data.cuda(non_blocking=True)

    def current_ep(self):
        return self.losslog.current_epoch

    def calculate_loss(self, *inputs):
        # --- Return losses in this way ---
        self.temp_loss = {loss: 0 for loss in self.loss_terms}
        return {pred: None for pred in self.pred_terms}
    
    def update(self):
        for i, loss in enumerate(self.loss_optimizer.keys(), start=1):
            self.losslog.loss[loss].optimizer.zero_grad()
            if i != len(self.loss_optimizer):
                self.temp_loss[loss].backward(retain_graph=True)
            else:
                self.temp_loss[loss].backward()
            self.losslog.loss[loss].optimizer.step()

    @timer
    def train(self, train_module=None, eval_module=None, early_stop=True, lr_decay=True, *args, **kwargs):
        self.early_stopping = EarlyStopping(*args, **kwargs)
        
        if not train_module:
            train_module = list(self.models.keys())
        params = [{'params': self.models[model].parameters()} for model in train_module]
        if self.extra_params.updatable:
            for param, value in self.extra_params.params.items():
                params.append({'params': value})
                
        for loss, [optimizer, lr] in self.loss_optimizer.items():
            self.losslog.loss[loss].set_optimizer(optimizer, lr, params)
            
        print(f"=========={self.notion} Training starting==========")

        # ===============train and validate each epoch==============
        train_range = range(1000) if early_stop else range(self.epochs)
        for epoch, _ in enumerate(train_range, start=1):
            # =====================train============================
            print('')
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
                        data_[key] = self.data_to_device(value)

                PREDS = self.calculate_loss(data_)
                self.update()

                for key in EPOCH_LOSS.keys():
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

                if idx % 5 == 0:
                    print(f"\r{self.name} train: epoch={epoch}/{train_range[-1]}, "
                          f"batch={idx}/{len(self.dataloader['train'])}, "
                          f"loss={self.temp_loss['LOSS'].item():.4f}, "
                          f"current best valid loss={self.best_val_loss:.4f} @ epoch {self.best_vloss_ep}    ", flush=True, end='')

            for key, value in EPOCH_LOSS.items():
                EPOCH_LOSS[key] = np.average(value)
            self.losslog('train', EPOCH_LOSS)
            self.extra_params.update()

            # =====================valid============================
            print('')
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
                        data_[key] = self.data_to_device(value)

                with torch.no_grad():
                    PREDS = self.calculate_loss(data_)
                for key in EPOCH_LOSS.keys():
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

                val_loss = np.average(EPOCH_LOSS['LOSS'])
                if 0 < val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_vloss_ep = self.current_ep()

                if idx % 5 == 0:
                    print(f"\r{self.name} valid: epoch={epoch}/{train_range[-1]}, "
                          f"batch={idx}/{len(self.dataloader['valid'])}, "
                          f"loss={self.temp_loss['LOSS'].item():.4f}, "
                          f"current best valid loss={self.best_val_loss:.4f} @ epoch {self.best_vloss_ep}        ", flush=True, end='')

                with open(f"{self.save_path}{self.name}_trained.txt", 'w') as logfile:
                    logfile.write(f"{self.notion}_{self.name}\n"
                                  f"Total epochs = {self.current_ep()}\n"
                                  f"Best : val_loss={self.best_val_loss} @ epoch {self.best_vloss_ep}\n"
                                  f"Modules:\n{list(self.models.values())}\n"
                                  )

            self.early_stopping(self.best_val_loss, early_stop, lr_decay)
            if lr_decay and self.early_stopping.decay_flag:
                self.losslog.decay(0.5)
                    
            if early_stop and self.early_stopping.stop_flag:
                self.losslog.in_training = False
                if 'save_model' in kwargs.keys() and kwargs['save_model'] is False:
                    break
                else:
                    print(f"\033[32mEarly Stopping triggered. Saving @ epoch {epoch}...\033[0m")
                    for model in train_module:
                        torch.save(self.models[model].state_dict(),
                                   f"{self.save_path}{self.name}_{model}_best.pth")
                    break
            print('')
            for key in EPOCH_LOSS.keys():
                EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
            self.losslog('valid', EPOCH_LOSS)

        if self.thread == 'multi':
            dist.destroy_process_group()
        return self.models

    @timer
    def test(self, test_module=None, loader: str = 'test', *args, **kwargs):

        if not test_module:
            test_module = list(self.models.keys())
        for model in test_module:
            self.models[model].eval()

        EPOCH_LOSS = {loss: [] for loss in self.loss_terms}
        self.losslog.reset('test', 'pred', dataset=loader)
        
        print(f"=========={self.notion} Test starting==========\n")

        for idx, data in enumerate(self.dataloader[loader], 0):
            data_ = {}
            length = 0
            for key, value in data.items():
                if key in self.modality:
                    data_[key] = self.data_to_device(value)
                    length = len(value)

            with torch.no_grad():
                for sample in range(length):
                    data_i = {key: data_[key][sample][np.newaxis, ...] for key in data_.keys()}
                    PREDS = self.calculate_loss(data_i)

                    for key in EPOCH_LOSS.keys():
                        EPOCH_LOSS[key].append(self.temp_loss[key].item())

                    self.losslog('pred', PREDS)

            if idx % 5 == 0:
                print(f"\r{self.name} test: sample={idx}/{len(self.dataloader[loader])}, "
                      f"loss={self.temp_loss['LOSS'].item():.4f}    ", end='', flush=True)

        self.losslog('test', EPOCH_LOSS)
        for key in EPOCH_LOSS.keys():
            EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])

        print(f"\nTest finished. Average loss={EPOCH_LOSS}")

    def plot_train_loss(self, title=None, double_y=False, plot_terms='all', autosave=False, **kwargs):
        fig = self.losslog.plot_train(title, plot_terms, double_y)
        if autosave:
            for filename, fig in fig.items():
                fig.savefig(f"{self.save_path}{filename}")

    def plot_test(self, select_inds=None, select_num=8, autosave=False, **kwargs):
        # According to actual usages
        self.losslog.generate_indices(select_inds, select_num)
        pass

    def save(self):
        print("Saving models...")
        for modelname, model in self.models.items():
            print(f"Saving {modelname}...")
            torch.save(model.state_dict(),
                       f"{self.save_path}{self.name}_{modelname}@ep{self.current_ep()}.pth")
        print("All saved!")

    def schedule(self, autosave=True, *args, **kwargs):
        # Training, testing and saving
        model = self.train(autosave=autosave, notion=self.notion, *args, **kwargs)
        self.plot_train_loss(autosave=autosave)
        self.test(loader='train')
        self.plot_test(select_num=8, autosave=autosave)
        self.test(loader='test')
        self.plot_test(select_num=8, autosave=autosave)
        self.losslog.save('preds', self.save_path)
        print(f'\n\033[32m{self.name} schedule Completed!\033[0m')
        return model

