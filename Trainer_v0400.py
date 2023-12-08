import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
from Loss import MyLoss


def reparameterize(mu, logvar):
    eps = torch.randn_like(mu)
    return mu + eps * torch.exp(logvar / 2)


class CsiEncoder(nn.Module):
    def __init__(self, latent_dim=16, feature_length=512):
        super(CsiEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.feature_length = feature_length

        self.cnn = nn.Sequential(
            # 2 * 90 * 100
            nn.Conv2d(2, 16, kernel_size=3, stride=(3, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            # 16 * 30 * 98
            nn.Conv2d(16, 64, kernel_size=3, stride=(2, 2), padding=0),
            nn.LeakyReLU(inplace=True),
            # 64 * 14 * 48
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            # 128 * 12 * 46
            nn.Conv2d(128, 256, kernel_size=3, stride=(1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            # 256 * 10 * 44
            nn.Conv2d(256, self.feature_length, kernel_size=3, stride=(1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            # 512 * 8 * 42
        )

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 42))

        self.lstm = nn.Sequential(
            nn.LSTM(self.feature_length, 2 * self.latent_dim, 2, batch_first=True, dropout=0.1),
        )

    def __str__(self):
        return 'CsiEnV0400'

    def forward(self, x):
        out = self.cnn(x)
        out = self.gap(out)
        out, (final_hidden_state, final_cell_state) = self.lstm.forward(
            out.view(-1, self.feature_length, 42).transpose(1, 2))

        out = out[:, -1, :]

        mu, logvar = out.view(-1, 2 * self.latent_dim).chunk(2, dim=-1)
        z = reparameterize(mu, logvar)

        return out, z, mu, logvar


class CsiDecoder(nn.Module):
    def __init__(self, latent_dim=16):
        super(CsiDecoder, self).__init__()

        self.latent_dim = latent_dim

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 18000),
        )

    def __str__(self):
        return 'CsiDeV0400'

    def forward(self, x):
        out = self.fclayers(x)
        return out.view(-1, 2, 90, 100)


class LatentEnTranslator(nn.Module):
    def __init__(self, latent_dim=16, repres_dim=128):
        super(LatentEnTranslator, self).__init__()

        self.latent_dim = latent_dim
        self.repres_dim = repres_dim

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.repres_dim),
            nn.ReLU(),
        )

    def __str__(self):
        return 'LatEnTrV0400'

    def forward(self, x):
        out = self.fclayers(x)

        return out


class LatentDeTranslator(nn.Module):
    def __init__(self, latent_dim=16, repres_dim=128):
        super(LatentDeTranslator, self).__init__()

        self.latent_dim = latent_dim
        self.repres_dim = repres_dim

        self.fclayers = nn.Sequential(
            nn.Linear(self.repres_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim),
            nn.ReLU(),
        )

    def __str__(self):
        return 'LatDeTrV0400'

    def forward(self, x):
        out = self.fclayers(x)

        return out


class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=16, active_func=nn.Tanh()):
        super(ImageEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.active_func = active_func

        self.cnn = nn.Sequential(
            # 1 * 128 * 128
            nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            # 128 * 64 * 64
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            # 128 * 32 * 32
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            # 128 * 16 * 16
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            # 256 * 8 * 8
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            # 256 * 4 * 4
        )

        self.fclayers = nn.Sequential(
            nn.Linear(4 * 4 * 256, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2 * self.latent_dim),
            self.active_func
        )

    def __str__(self):
        return 'ImgEnV0400'

    def forward(self, x):
        out = self.cnn(x)

        if self.bottleneck == 'fc':
            out = self.fclayers(out.view(-1, 4 * 4 * 256))
        elif self.bottleneck == 'gap':
            out = self.gap(out)
            out = nn.Sigmoid(out)

        mu, logvar = out.view(-1, 2 * self.latent_dim).chunk(2, dim=-1)
        z = reparameterize(mu, logvar)

        return out, z, mu, logvar


class ImageDecoder(nn.Module):
    def __init__(self, latent_dim=16):
        super(ImageDecoder, self).__init__()

        self.latent_dim = latent_dim

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU()
        )

        self.cnn = nn.Sequential(
            # 128 * 4 * 4
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            # 128 * 8 * 8
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            # 128 * 16 * 16
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            # 128 * 32 * 32
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            # 128 * 64 * 64
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            self.active_func,
            # 1 * 128 * 128
        )

    def __str__(self):
        return 'ImgDeV0400' + self.bottleneck.capitalize()

    def forward(self, x):

        out = self.fclayers(x.view(-1, self.latent_dim))
        out = self.cnn(out.view(-1, 256, 1, 1))
        return out.view(-1, 1, 128, 128)


class Trainer:
    def __init__(self, train_loader, valid_loader, test_loader, lr=1e-4, epochs=10, cuda=1):
        self.models = self.__gen_models__()
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda:" + str(cuda) if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.recon_lossfun = nn.MSELoss(reduction='sum')
        self.beta = 1.2

        self.temp_loss = {}
        self.loss = {'csi': MyLoss(loss_terms=['LOSS', 'KL', 'RECON'], pred_terms=['GT', 'PRED', 'IND']),
                     'img': MyLoss(loss_terms=['LOSS', 'KL', 'RECON'], pred_terms=['GT', 'PRED', 'IND']),
                     'inta': MyLoss(loss_terms=['LOSS', 'CSI_L', 'IMG_L', 'INTA'],
                                    pred_terms=['CSI_L', 'RE_CSI_L', 'IMG_L', 'RE_IMG_L', 'CSI_R', 'IMG_R', 'IND'])
                     }

    @staticmethod
    def __gen_models__():
        csien = CsiEncoder()
        cside = CsiDecoder()
        imgen = ImageEncoder()
        imgde = ImageDecoder()
        csilen = LatentEnTranslator()
        csilde = LatentDeTranslator()
        imglen = LatentEnTranslator()
        imglde = LatentDeTranslator()
        return {'csien': csien,
                'cside': cside,
                'imgen': imgen,
                'imgde': imgde,
                'csilen': csilen,
                'csilde': csilde,
                'imglen': imglen,
                'imglde': imglde
                }

    def vae_loss(self, pred, gt, mu, logvar):
        recon_loss = self.recon_lossfun(pred, gt) / pred.shape[0]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss * self.beta
        return loss, kl_loss, recon_loss

    def calculate_csi_loss(self, x, i=None):
        latent, z, mu, logvar = self.models['csien'](x)
        recon_csi = self.models['cside'](z)
        loss, kl_loss, recon_loss = self.vae_loss(recon_csi, x, mu, logvar)
        self.temp_loss = {'LOSS': loss,
                          'KL': kl_loss,
                          'RECON': recon_loss}
        return {'GT': x,
                'PRED': recon_csi,
                'IND': i}

    def calculate_img_loss(self, y, i=None):
        latent, z, mu, logvar = self.models['imgen'](y)
        recon_img = self.models['imgde'](z)
        loss, kl_loss, recon_loss = self.vae_loss(recon_img, y, mu, logvar)
        self.temp_loss = {'LOSS': loss,
                          'KL': kl_loss,
                          'RECON': recon_loss}
        return {'GT': y,
                'PRED': recon_img,
                'IND': i}

    def train_csi_inner(self, autosave=False, notion=''):
        optimizer = self.optimizer([{'params': self.models['csien'].parameters()},
                                    {'params': self.models['cside'].parameters()}], lr=self.lr)
        self.loss['csi'].logger(self.lr, self.epochs)
        for epoch in range(self.epochs):
            # =====================train============================
            self.models['csien'].train()
            self.models['cside'].train()
            EPOCH_LOSS = {'LOSS': [],
                          'KL': [],
                          'RECON': []}
            for idx, (data_x, data_y, index) in enumerate(self.train_loader, 0):
                data_x = data_x.to(torch.float32).to(self.device)
                optimizer.zero_grad()

                PREDS = self.calculate_csi_loss(x=data_x)
                self.temp_loss['LOSS'].backward()
                optimizer.step()
                for key in EPOCH_LOSS.keys():
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

                if idx % (len(self.train_loader) // 5) == 0:
                    print(f"\rCSI: epoch={epoch}/{self.epochs}, batch={idx}/{len(self.train_loader)}, "
                          f"loss={self.temp_loss['LOSS'].item()}", end='')

            for key in EPOCH_LOSS.keys():
                EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
            self.loss['csi'].update('train', EPOCH_LOSS)

            # =====================valid============================
            self.models['csien'].eval()
            self.models['cside'].eval()
            EPOCH_LOSS = {'LOSS': [],
                          'KL': [],
                          'RECON': []}

            for idx, (data_x, data_y, index) in enumerate(self.valid_loader, 0):
                csi = data_x.to(torch.float32).to(self.device)
                with torch.no_grad():
                    PREDS = self.calculate_csi_loss(x=csi)
                for key in EPOCH_LOSS.keys():
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())
            for key in EPOCH_LOSS.keys():
                EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
            self.loss['csi'].update('valid', EPOCH_LOSS)

    def train_img_inner(self, autosave=False, notion=''):
        optimizer = self.optimizer([{'params': self.models['imgen'].parameters()},
                                    {'params': self.models['imgde'].parameters()}], lr=self.lr)
        self.loss['img'].logger(self.lr, self.epochs)
        for epoch in range(self.epochs):
            # =====================train============================
            self.models['imgen'].train()
            self.models['imgde'].train()
            EPOCH_LOSS = {'LOSS': [],
                          'KL': [],
                          'RECON': []}
            for idx, (data_x, data_y, index) in enumerate(self.train_loader, 0):
                data_y = data_y.to(torch.float32).to(self.device)
                optimizer.zero_grad()

                PREDS = self.calculate_img_loss(y=data_y)
                self.temp_loss['LOSS'].backward()
                optimizer.step()
                for key in EPOCH_LOSS.keys():
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

                if idx % (len(self.train_loader) // 5) == 0:
                    print(f"\rIMG: epoch={epoch}/{self.epochs}, batch={idx}/{len(self.train_loader)}, "
                          f"loss={self.temp_loss['LOSS'].item()}", end='')

            for key in EPOCH_LOSS.keys():
                EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
            self.loss['img'].update('train', EPOCH_LOSS)

            # =====================valid============================
            self.models['imgen'].eval()
            self.models['imgde'].eval()
            EPOCH_LOSS = {'LOSS': [],
                          'KL': [],
                          'RECON': []}

            for idx, (data_x, data_y, index) in enumerate(self.valid_loader, 0):
                img = data_y.to(torch.float32).to(self.device)
                with torch.no_grad():
                    PREDS = self.calculate_img_loss(y=img)
                for key in EPOCH_LOSS.keys():
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())
            for key in EPOCH_LOSS.keys():
                EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
            self.loss['img'].update('valid', EPOCH_LOSS)

    def test_csi_inner(self, mode='test'):
        self.models['csien'].eval()
        self.models['cside'].eval()
        EPOCH_LOSS = {'LOSS': [],
                      'KL': [],
                      'RECON': []}

        if mode == 'test':
            loader = self.test_loader
        elif mode == 'train':
            loader = self.train_loader

        for idx, (data_x, data_y, index) in enumerate(loader, 0):
            data_x = data_x.to(torch.float32).to(self.device)
            with torch.no_grad():
                for sample in range(loader.batch_size):
                    ind = index[sample][np.newaxis, ...]
                    csi = data_x[sample][np.newaxis, ...]
                    PREDS = self.calculate_csi_loss(x=csi, i=ind)

                    for key in EPOCH_LOSS.keys():
                        EPOCH_LOSS[key].append(self.temp_loss[key].item())

                    self.loss['csi'].update('pred', PREDS)

            if idx % (len(loader)//5) == 0:
                print(f"\rCSI: test={idx}/{len(loader)}, loss={self.temp_loss['LOSS'].item()}", end='')

        self.loss['csi'].update('test', EPOCH_LOSS)
        for key in EPOCH_LOSS.keys():
            EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
        print(f"\nTest finished. Average loss={EPOCH_LOSS}")

    def test_img_inner(self, mode='test'):
        self.models['imgen'].eval()
        self.models['imgde'].eval()
        EPOCH_LOSS = {'LOSS': [],
                      'KL': [],
                      'RECON': []}

        if mode == 'test':
            loader = self.test_loader
        elif mode == 'train':
            loader = self.train_loader

        for idx, (data_x, data_y, index) in enumerate(loader, 0):
            data_y = data_y.to(torch.float32).to(self.device)
            with torch.no_grad():
                for sample in range(loader.batch_size):
                    ind = index[sample][np.newaxis, ...]
                    img = data_y[sample][np.newaxis, ...]
                    PREDS = self.calculate_img_loss(y=img, i=ind)

                    for key in EPOCH_LOSS.keys():
                        EPOCH_LOSS[key].append(self.temp_loss[key].item())

                    self.loss['img'].update('pred', PREDS)

            if idx % (len(loader)//5) == 0:
                print(f"\rIMG: test={idx}/{len(loader)}, loss={self.temp_loss['LOSS'].item()}", end='')

        self.loss['img'].update('test', EPOCH_LOSS)
        for key in EPOCH_LOSS.keys():
            EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
        print(f"\nTest finished. Average loss={EPOCH_LOSS}")

    def calculate_outer_loss(self, x, y, i=None):
        with torch.no_grad():
            latent_c, z_c, mu_c, logvar_c = self.models['csien'](x)
            latent_i, z_i, mu_i, logvar_i = self.models['imgen'](y)
        repr_c = self.models['csilen'](z_c)
        repr_i = self.models['imglen'](z_i)
        recon_z_c = self.models['csilde'](repr_c)
        recon_z_i = self.models['imglde'](repr_i)
        csil_loss = self.recon_lossfun(recon_z_c, z_c)
        imgl_loss = self.recon_lossfun(recon_z_i, z_i)
        inta_loss = self.recon_lossfun(repr_c, repr_i)
        loss = csil_loss + imgl_loss + inta_loss

        self.temp_loss = {'LOSS': loss,
                          'CSI_L': csil_loss,
                          'IMG_L': imgl_loss,
                          'INTA': inta_loss}
        return {'CSI_L': z_c,
                'RE_CSI_L': recon_z_c,
                'IMG_L': z_i,
                'RE_IMG_L': recon_z_i,
                'CSI_R': repr_c,
                'IMG_R': repr_i,
                'IND': i}

    def train_outer(self):
        optimizer = self.optimizer([{'params': self.models['csilen'].parameters()},
                                    {'params': self.models['csilde'].parameters()},
                                    {'params': self.models['imglen'].parameters()},
                                    {'params': self.models['imglde'].parameters()},
                                    ], lr=self.lr)
        self.loss['inta'].logger(self.lr, self.epochs)

        for epoch in range(self.epochs):

            # =====================train============================
            self.models['csien'].eval()
            self.models['imgen'].eval()
            self.models['csilen'].train()
            self.models['csilde'].train()
            self.models['imglen'].train()
            self.models['imglde'].train()
            EPOCH_LOSS = {'LOSS': [],
                          'CSI_L': [],
                          'IMG_L': [],
                          'INTA': []}

            for idx, (data_x, data_y, index) in enumerate(self.train_loader, 0):
                data_x = data_x.to(torch.float32).to(self.device)
                data_y = data_y.to(torch.float32).to(self.device)

                PREDS = self.calculate_outer_loss(data_x, data_y, index)
                optimizer.zero_grad()
                self.temp_loss['LOSS'].backward()
                optimizer.step()

                for key in EPOCH_LOSS.keys():
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

                if idx % (len(self.train_loader) // 5) == 0:
                    print(f"\rIntact: epoch={epoch}/{self.epochs}, batch={idx}/{len(self.train_loader)}, "
                          f"loss={self.temp_loss['LOSS'].item()}", end='')

            for key in EPOCH_LOSS.keys():
                EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
            self.loss['inta'].update('train', EPOCH_LOSS)

            # =====================valid============================
            self.models['csien'].eval()
            self.models['imgen'].eval()
            self.models['csilen'].eval()
            self.models['csilde'].eval()
            self.models['imglen'].eval()
            self.models['imglde'].eval()
            EPOCH_LOSS = {'LOSS': [],
                          'CSI_L': [],
                          'IMG_L': [],
                          'INTA': []}

            for idx, (data_x, data_y, index) in enumerate(self.valid_loader, 0):
                data_x = data_x.to(torch.float32).to(self.device)
                data_y = data_y.to(torch.float32).to(self.device)
                with torch.no_grad():
                    PREDS = self.calculate_outer_loss(data_x, data_y, index)

                for key in EPOCH_LOSS.keys():
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

            for key in EPOCH_LOSS.keys():
                EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
            self.loss['inta'].update('valid', EPOCH_LOSS)

    def test_outer(self, mode='test'):

        self.models['csien'].eval()
        self.models['imgen'].eval()
        self.models['csilen'].eval()
        self.models['csilde'].eval()
        self.models['imglen'].eval()
        self.models['imglde'].eval()
        EPOCH_LOSS = {'LOSS': [],
                      'CSI_L': [],
                      'IMG_L': [],
                      'INTA': []}

        if mode == 'test':
            loader = self.test_loader
        elif mode == 'train':
            loader = self.train_loader

        for idx, (data_x, data_y, index) in enumerate(loader, 0):
            data_x = data_x.to(torch.float32).to(self.device)
            data_y = data_y.to(torch.float32).to(self.device)
            with torch.no_grad():
                for sample in range(loader.batch_size):
                    ind = index[sample][np.newaxis, ...]
                    csi = data_x[sample][np.newaxis, ...]
                    img = data_y[sample][np.newaxis, ...]
                    PREDS = self.calculate_outer_loss(csi, img, ind)

                for key in EPOCH_LOSS.keys():
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

                self.loss['inta'].update('plot', PREDS)

            if idx % (len(loader) // 5) == 0:
                print(f"\rINTA: test={idx}/{len(loader)}, loss={self.temp_loss['LOSS'].item()}", end='')

        self.loss['inta'].update('test', EPOCH_LOSS)
        for key in EPOCH_LOSS.keys():
            EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
        print(f"\nTest finished. Average loss={EPOCH_LOSS}")


if __name__ == "__main__":
    IMG = (1, 1, 128, 128)
    CSI = (2, 90, 100)
    LAT = (1, 16)
    REPR = (1, 128)

    m = CsiEncoder()
    summary(m, input_size=CSI)

