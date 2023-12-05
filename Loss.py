class MyLoss:
    def __init__(self, loss_terms, plot_terms):
        self.loss = {'train': {term: [] for term in loss_terms},
                     'valid': {term: [] for term in loss_terms},
                     'test': {term: [] for term in plot_terms}
                     }
        self.lr = []
        self.epochs = [0]

    def logger(self, lr, epochs):

        # First round
        if not self.lr:
            self.lr.append(lr)
            self.epochs.append(epochs)
        else:
            # Not changing learning rate
            if lr == self.lr[-1]:
                self.epochs[-1] += epochs

            # Changing learning rate
            if lr != self.lr[-1]:
                last_end = self.epochs[-1]
                self.lr.append(lr)
                self.epochs.append(last_end + epochs)

    def update(self, mode, losses):
        if mode in ('train', 'valid', 'test'):
            for key in losses.keys():
                self.loss[mode][key].append(losses[key])

        if mode == 'plot':
            for key in losses.keys():
                self.loss['test'][key].append(losses[key].cpu().detach().numpy().squeeze())

