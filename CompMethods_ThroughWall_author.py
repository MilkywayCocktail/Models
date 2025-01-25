import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
import os
from Trainer import BasicTrainer, TrainingPhase, ValidationPhase
from Loss import MyLossLog
from Model import *

##############################################################################
# -------------------------------------------------------------------------- #
# Notes
#
# 1) Wi2Vi uses 56x3x3x29 CSI, while we use 30x3x3x100
#
# 2) Video frames are aligned with the first packets of CSI
#
# 3) Wi2Vi video FPS = 30 -> 6, CSI rate = 100Hz
#
# 4) Wi2Vi train:test = 95:5
#
# 5) Wi2Vi lr=2e-3 and lower; epoch=1000; batch size=32
#
# 6) Wi2Vi outputs 320x240 images
# -------------------------------------------------------------------------- #
##############################################################################

version = 'ThroughWallAuth'

import numpy as np
import wandb
import math
import torch
import torch.nn as nn
from torch.distributions import Normal as _Normal
from itertools import combinations
from scipy.stats import norm
import pytorch_lightning as L
from torchmetrics.image.fid import FrechetInceptionDistance

EPS = 1e-8

# set numpy seed
np.random.seed(0)

class Preprocess:
    def __init__(self, new_size=(128, 128)):
        self.new_size = new_size
        self.batch_size = 32

    def transform(self, tensor):
        ten =  F.interpolate(tensor, size=self.new_size, mode='bilinear', align_corners=False)
        # No need to normalize because the depth images are already [0, 1]
        return ten
    
    @staticmethod
    def encode_time(x, L=10, window_size=151):
        window_size *= 3
        frequencies = np.array([2**i for i in range(L)])
        x = x / window_size
        pos_enc = np.concatenate([torch.sin(frequencies[:, None] * torch.pi * x),
                                torch.cos(frequencies[:, None] * torch.pi * x)], axis=0)
        return pos_enc
    
    def __call__(self, data, modalities):
        """
        Preprocess after retrieving data
        """
        
        #  Transform images
        if self.new_size and 'rimg' in modalities:
            data['rimg'] = self.transform(data['rimg'])

        if 'csi' in modalities:
            data['csi'] = torch.abs(data['csi']) # 3 Rx
            
        if 'csitime' in modalities:
            data['csitime'] = self.encode_time(data['csitime'])

        return data


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)
    

# Image Variational Autoencoder
class ImageVAE(nn.Module):

    def __init__(self,
                 in_channels = 1,
                 time_input_dim = 20,
                 z_dim = 128,
                 hidden_dims: [int] = None,
                 beta: int = 4,
                 aggregation: str = 'concat') -> None:
        super(ImageVAE, self).__init__()

        self.aggregation = aggregation

        self.latent_dim = z_dim
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [48, 96, 128, 192, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()),
            )
            in_channels = h_dim

        self.image_encoder = nn.Sequential(*modules)
        self.time_encoder = MLP(input_dim=time_input_dim,
                                hidden_dim=time_input_dim//2,
                                output_dim=time_input_dim//4)
        
        self.latent_encoder = nn.Linear(hidden_dims[-1]*4 + time_input_dim//4, z_dim * 2) # 2048 + 27 -> 256
        
        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(z_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1,kernel_size=3, padding=1),
            nn.Tanh())
        
        tmp_noise_par = torch.empty((1, 3, 128, 128)).fill_(-3) 
        self.logvar_out = nn.Parameter(data=tmp_noise_par, requires_grad=True)

    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        time, image = x['csitime'], x['rimg']
        image = self.image_encoder(image)
        # time = self.time_encoder(time.reshape(time.shape[0],1,1,-1))
        image = torch.flatten(image, start_dim=1)
        time = time[:, 0, 0, :]
        x = torch.concat([time, image], dim=1)
        mu, logvar = self.latent_encoder(x).chunk(2, dim=-1)

        # clamp logvar to -10 and 10 to avoid numerical instability
        logvar = torch.clamp(logvar, -10, 10)

        return mu, logvar

    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result.view(-1, 512, 2, 2))
        result = self.final_layer(result)
        s = torch.exp(0.5 * self.logvar_out)
        ll = Normal(loc=result, scale=s)
        return ll, result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)[0], mu, logvar
    

# CSI Variational Autoencoder
class CSIVAE(nn.Module):
    def __init__(self,
                 feature_input_dim = 151 * 3,
                 time_input_dim = 20,
                 sequence_length = 151,
                 z_dim = 128,
                 aggregation_method: str='concat'):
        super(CSIVAE, self).__init__()

        self.aggregate_method = aggregation_method

        output = 64

        self.feature_encoder = MLP(input_dim=feature_input_dim,hidden_dim=feature_input_dim//2,output_dim=output)

        self.time_encoder = MLP(input_dim=time_input_dim,hidden_dim=time_input_dim//8,output_dim=time_input_dim//16)
        
        if self.aggregate_method=='concat':
            self.latent_encoder = MLP(input_dim=output*sequence_length+time_input_dim//16,hidden_dim=z_dim*2,output_dim=z_dim*2)
            
        # elif self.aggregate_method=='gaussian':
        #     self.latent_encoder = MLP(input_dim=output+time_input_dim//16,hidden_dim=z_dim*2,output_dim=z_dim*2)
        #     gaussian = norm.pdf(np.arange(sequence_length),loc=sequence_length / 2, scale=sequence_length / 2)
        #     weights = gaussian / gaussian.sum()
        #     self.weighting = torch.nn.Parameter(torch.Tensor(weights.reshape((1, 1, -1, 1))),requires_grad=False)
        #     self.sum_weight = self.weighting.sum()

        # elif self.aggregate_method=='uniform':
        #     self.latent_encoder = MLP(input_dim=output+time_input_dim//16,hidden_dim=z_dim*2,output_dim=z_dim*2)
        #     uniform = np.ones(sequence_length)
        #     weights = uniform / uniform.sum()
        #     self.weighting = torch.nn.Parameter(torch.Tensor(weights.reshape((1, 1, -1, 1))),requires_grad=False)
        #     self.sum_weight = self.weighting.sum()

        self.latent_decoder = MLP(z_dim, z_dim*2, feature_input_dim)
        tmp_noise_par = torch.empty((1, feature_input_dim)).fill_(-3)
        self.logvar_out = nn.Parameter(data=tmp_noise_par, requires_grad=True)

    def aggregate_features(self, features):
        return ((features*self.weighting).sum(dim=2)/self.sum_weight).squeeze()
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):

        time, feature = x['csitime'], x['csi']

        # swap sequence dim with feature dim
        # CSI = 32 * 151 * 30 * 3
        feature = feature.permute(0, 2, 1, 3)
        feature = self.feature_encoder(feature.view(32, 30, -1))
        time = self.time_encoder(time.reshape(time.shape[0],1,1,-1))
 
        # feature aggregation
        if self.aggregate_method=='concat':
            feature = feature.reshape(feature.shape[0], -1)
        # elif self.aggregate_method=='gaussian':
        #     feature = self.aggregate_features(feature)
        # elif self.aggregate_method=='uniform':
        #     feature = self.aggregate_features(feature)
        
        # concatenate features and temporal encoding
        time = time.view(time.shape[0], -1)
        x = torch.concat([time, feature], dim=1)
        
        mu, logvar = self.latent_encoder(x).chunk(2, dim=-1)

        # clamp logvar to -10 and 10 to avoid numerical instability
        logvar = torch.clamp(logvar, -10, 10)

        return mu, logvar
    
    def decode(self, z):
        result = self.latent_decoder(z)
        ll = Normal(loc=result, scale=torch.exp(0.5 * self.logvar_out))
        return ll, result

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    

class MoPoEVAE(nn.Module):
    r"""
    Mixture-of-Product-of-Experts Variational Autoencoder.

    Code is based on: https://github.com/thomassutter/MoPoE

    Args:
        cfg (str): Path to configuration file. Model specific parameters in addition to default parameters:

            - model.beta (int, float): KL divergence weighting term.
            - encoder.default._target_ (multiviewae.architectures.mlp.VariationalEncoder): Type of encoder class to use.
            - encoder.default.enc_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Encoding distribution.
            - decoder.default._target_ (multiviewae.architectures.mlp.VariationalDecoder): Type of decoder class to use.
            - decoder.default.init_logvar (int, float): Initial value for log variance of decoder.
            - decoder.default.dec_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Decoding distribution.

        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.

    References
    ----------
    Sutter, Thomas & Daunhawer, Imant & Vogt, Julia. (2021). Generalized Multimodal ELBO.
    """

    def __init__(
        self,
        imgMean=0,
        imgStd=1,
        weight_ll=False,
        sequence_length=151,
        z_dim=128,
        frequence_L=10,
        aggregate_method='concat',
        log=None,
        training=True
    ):
        super(MoPoEVAE, self).__init__()
        self.weight_ll = weight_ll

        self.random_index = 0
        self.prior_mean = torch.nn.Parameter(torch.zeros(z_dim), requires_grad=False)
        self.prior_logvar = torch.nn.Parameter(torch.zeros(z_dim), requires_grad=False)
        self.prior = Normal(loc=self.prior_mean, logvar=self.prior_logvar)
        self.subsets = self.set_subsets()
        self.beta = 1
        self.logImage = log
        self.imgMean = torch.tensor(imgMean.reshape(1, 1, 1, 1)).to(self.device)
        self.imgStd = torch.tensor(imgStd.reshape(1, 1, 1, 1)).to(self.device)
        self.training = training

        if self.weight_ll:
            self.ll_weighting = 1/2
        else:
            self.ll_weighting = 1
            
        self.encoders = nn.ModuleList([
            CSIVAE(
                feature_input_dim=30 * 3,
                time_input_dim=frequence_L*2,
                sequence_length=sequence_length, z_dim=z_dim, aggregation_method=aggregate_method
            ),
            ImageVAE(
                in_channels=1,
                time_input_dim=frequence_L*2,
                z_dim=z_dim
            ),
        ])

    def encode(self, x):
        r"""Forward pass through encoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): list containing the MoE joint encoding distribution. If training, the model also returns the encoding distribution for each subset. 
        """
        mu = []
        logvar = []

        for inpt, encoder in zip(x, self.encoders):
            mu_, logvar_ = encoder.encode(inpt)
            mu.append(mu_)
            logvar.append(logvar_)

        mu = torch.stack(mu)
        logvar = torch.stack(logvar)

        mu_out = []
        logvar_out = []

        qz_xs = []
        for subset in self.subsets:
            mu_s = mu[subset]
            logvar_s = logvar[subset]
            if len(subset) == 2:
                mu_ = self.prior.loc
                mu_ = mu_.expand(mu[0].shape).to(mu[0].device)
                logvar_ = torch.log(self.prior.variance).to(mu[0].device)
                logvar_ = logvar_.expand(logvar[0].shape)
                mu_ = mu_.unsqueeze(0)
                logvar_ = logvar_.unsqueeze(0)
                mu_s = torch.cat([mu_s, mu_], dim=0)
                logvar_s = torch.cat([logvar_s, logvar_], dim=0)

            mu_s, logvar_s = ProductOfExperts()(mu_s, logvar_s)
            mu_out.append(mu_s)
            logvar_out.append(logvar_s)
            qz_x = Normal(loc=mu_s, logvar=logvar_s)
            qz_xs.append(qz_x)
        mu_out = torch.stack(mu_out)
        logvar_out = torch.stack(logvar_out)

        moe_mu, moe_logvar = MixtureOfExperts()(mu_out, logvar_out)

        qz_x = Normal(loc=moe_mu, logvar=moe_logvar)
        return [qz_xs, qz_x]

    def encode_subset(self, x, subset):
        r""" Forward pass through encoder networks for a subset of modalities.
        Args:
            x (list): list of input data of type torch.Tensor.
            subset (list): list of modalities to encode.

        Returns:
            (list): list containing the MoE joint encoding distribution. 
        """
        mu = []
        logvar = []

        for i in subset:
            mu_, logvar_ = self.encoders[i].encode(x[i])
            mu.append(mu_)
            logvar.append(logvar_)

        if len(subset) == 2:
            mu_ = self.prior.loc
            mu_ = mu_.expand(mu[0].shape).to(mu[0].device)
            logvar_ = torch.log(self.prior.variance).to(mu[0].device)
            logvar_ = logvar_.expand(logvar[0].shape)
            mu.append(mu_)
            logvar.append(logvar_)

        mu = torch.stack(mu)
        logvar = torch.stack(logvar)

        mu, logvar = ProductOfExperts()(mu, logvar)
        qz_x = Normal(loc=mu, logvar=logvar)
        return [qz_x]

    def decode(self, qz_x):
        r"""Forward pass of joint latent dimensions through decoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): A nested list of decoding distributions, px_zs. The outer list has a single element indicating the shared latent dimensions. 
            The inner list is a n_view element list with the position in the list indicating the decoder index.
        """
        px_zs = []
        x_hats = []
        for i in range(2):
            px_z, x_hat = self.encoders[i].decode(qz_x[0]._sample(training=self.training))
            px_zs.append(px_z)
            x_hats.append(x_hat)
        return [px_zs], [x_hats]

    def forward(self, x):
        r"""Apply encode and decode methods to input data to generate the joint and subset latent dimensions and data reconstructions. 

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.
        """
        qz_xs, qz_x = self.encode(x)

        px_zs = self.decode([qz_x])
        fwd_rtn = {"px_zs": px_zs, "qz_xs_subsets": qz_xs, "qz_x_joint": qz_x}
        return fwd_rtn

    def loss_function(self, x, fwd_rtn):
        r"""Calculate MoPoE VAE loss.

        Args:
            x (list): list of input data of type torch.Tensor.
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.

        Returns:
            losses (dict): dictionary containing each element of the MoPoE VAE loss.
        """
        px_zs = fwd_rtn["px_zs"]
        qz_xs = fwd_rtn["qz_xs_subsets"]

        kl = self.calc_kl_moe(qz_xs)
        ll = self.calc_ll(x, px_zs)

        total = self.beta * kl - ll

        losses = {"loss": total, "kl": kl, 'll': ll}
        return losses

    def calc_kl_moe(self, qz_xs):
        r"""Calculate KL-divergence between the each PoE subset posterior and the prior distribution.

        Args:
            qz_xs (list): list of encoding distributions.

        Returns:
            (torch.Tensor): KL-divergence loss.
        """
        weight = 1/len(qz_xs)
        kl = 0
        for qz_x in qz_xs:
            kl += qz_x.kl_divergence(self.prior).mean(0).sum()
        return kl*weight

    def set_subsets(self):
        """Create combinations of subsets of views.

        Returns:
            subset_list (list): list of unique combinations of n_views.
        """
        n_views = 2
        xs = list(range(0, n_views))
        tmp = [list(combinations(xs, n+1)) for n in range(len(xs))]
        subset_list = [list(item) for sublist in tmp for item in sublist]
        return subset_list

    def calc_ll(self, x, px_zs):
        r"""Calculate log-likelihood loss.

        Args:
            x (list): list of input data of type torch.Tensor.
            px_zs (list): list of decoding distributions.

        Returns:
            ll (torch.Tensor): Log-likelihood loss.
        """
        ll = 0
        i = 1
        ll += px_zs[0][0][i].log_likelihood(self.encoders[i].aggregate_features(x[i][1].permute(0, 1, 3, 2)) if i == 0 else x[i][1]).mean(0).sum()*self.ll_weighting  
        return ll

    # FID update
    def update_metrics(self, batch):
        real = batch[1][1]
        fake = self.decode(self.encode_subset(batch, [0]))[1][0][1]

        # reverse normalization
        real = real * self.imgStd.to(self.device) + self.imgMean.to(self.device)
        fake = fake * self.imgStd.to(self.device) + self.imgMean.to(self.device)

        # clip to [0,1]
        real = torch.clamp(real, 0, 1)
        fake = torch.clamp(fake, 0, 1)
        
        # convert to unit8
        real = (real*255).type(torch.uint8)
        fake = (fake*255).type(torch.uint8)

        # update FID state
        self.FID.update(real, real=True)
        self.FID.update(fake, real=False)

    def log_image(self, batch, name):
        reconstruction_from_csi = self.decode(self.encode_subset(batch, [0]))[1][0][1]
        reconstruction_from_image = self.decode(self.encode_subset(batch, [1]))[1][0][1]
        cat = torch.cat((batch[1][1], reconstruction_from_image,reconstruction_from_csi), dim=-1)

        if self.logImage:
            self.logger.log_image(f'{name}', [wandb.Image(cat)], self.current_epoch)
  
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(
            lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate, amsgrad=True)
        return optimizer


def compute_log_alpha(mu, logvar):
    return (logvar - 2 * torch.log(torch.abs(mu) + 1e-8)).clamp(min=-8, max=8)

class MixtureOfExperts(nn.Module):
    """Return parameters for mixture of independent experts.
    Implementation from: https://github.com/thomassutter/MoPoE

    Args:
    mus (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvars (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mus, logvars):

        num_components = mus.shape[0]
        num_samples = mus.shape[1]
        weights = (1/num_components) * \
            torch.ones(num_components).to(mus[0].device)
        idx_start = []
        idx_end = []
        for k in range(0, num_components):
            if k == 0:
                i_start = 0
            else:
                i_start = int(idx_end[k-1])
            if k == num_components-1:
                i_end = num_samples
            else:
                i_end = i_start + int(torch.floor(num_samples*weights[k]))
            idx_start.append(i_start)
            idx_end.append(i_end)
        idx_end[-1] = num_samples

        mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(num_components)])
        logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(num_components)])

        return mu_sel, logvar_sel

class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvar (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """

    def forward(self, mu, logvar):
        var = torch.exp(logvar) + EPS
        T = 1. / (var + EPS)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + EPS)
        return pd_mu, pd_logvar

class Normal(_Normal):
    """Univariate normal distribution. Inherits from torch.distributions.Normal.

    Args:
        loc (int, torch.Tensor): Mean of distribution.
        scale (int, torch.Tensor): Standard deviation of distribution.
    """

    def __init__(
        self,
        **kwargs,
    ):
        self.loc = kwargs['loc']
        if 'logvar' in kwargs:
            self.logvar = kwargs['logvar']
            self.scale = kwargs['logvar'].mul(0.5).exp_()+EPS

        elif 'scale' in kwargs:
            self.scale = kwargs['scale']
            if not isinstance(self.scale, torch.Tensor):
                self.scale = torch.tensor(self.scale)
            self.logvar = 2 * torch.log(self.scale)
        super().__init__(loc=self.loc, scale=self.scale)

    @property
    def variance(self):
        return self.scale.pow(2)

    def kl_divergence(self, other):
        logvar0 = self.logvar
        mu0 = self.loc
        logvar1 = other.logvar
        mu1 = other.loc

        return -0.5 * (1 - logvar0.exp()/logvar1.exp() - (mu0-mu1).pow(2)/logvar1.exp() + logvar0 - logvar1)

    
    def sparse_kl_divergence(self):
        """
        Implementation from: https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn/blob/master/KL%20approximation.ipynb

        """
        mu = self.loc
        logvar = torch.log(self.variance)
        log_alpha = compute_log_alpha(mu, logvar)
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        neg_KL = (
            k1 * torch.sigmoid(k2 + k3 * log_alpha)
            - 0.5 * torch.log1p(torch.exp(-log_alpha))
            - k1
        )
        return -neg_KL
    

    def log_likelihood(self, x):
        return self.log_prob(x)

    def _sample(self, *kwargs, training=False, return_mean=True):
        if training:
            return self.rsample(*kwargs)

        if return_mean:
            return self.loc
        return self.sample()



class ThroughWallTrainer(BasicTrainer):
    
    def __init__(self,
                 beta=0.5,
                 recon_lossfunc=nn.BCEWithLogitsLoss(reduction='sum'),
                 *args, **kwargs
                 ):
        
        super(ThroughWallTrainer, self).__init__(*args, **kwargs)
    
        self.modality = {'rimg', 'csi', 'csitime', 'tag', 'ind'}

        self.beta = beta
        self.recon_lossfunc = recon_lossfunc
        
        self.loss_terms = ('LOSS', 'KL', 'RECON')
        self.pred_terms = ('GT', 'PRED', 'LAT', 'TAG', 'IND')
        
        self.losslog = MyLossLog(name=self.name,
                           loss_terms=self.loss_terms,
                           pred_terms=self.pred_terms)
        
        self.model = Model(device=self.device)
        self.models = vars(self.model)
        
        
    def kl_loss(self, mu, logvar, prior_mu=0, prior_logvar=0, weights=None):
        """
        Calculate the KL divergence between the learned distribution and the prior (standard normal).
        
        Parameters:
        - mu: Tensor of means for each expert in the mixture (batch_size x input_dim).
        - logvar: Tensor of log-variances for each expert in the mixture (batch_size x input_dim).
        - prior_mu: Mean of the prior (default 0 for standard normal).
        - prior_logvar: Log variance of the prior (default 0 for standard normal).
        - weights: Mixture weights for the experts (softmaxed logits, batch_size).
        
        Returns:
        - kl_loss: KL divergence between the mixture of experts and the prior.
        """
        
        # Standard normal distribution (mean=0, logvar=0)
        prior_var = torch.exp(prior_logvar)
        prior_std = torch.sqrt(prior_var)
        
        # Calculate variances for each expert in the mixture
        var = torch.exp(logvar)  # Variance of the current expert
        
        # KL divergence for each expert, given prior
        kl_experts = 0.5 * (logvar - prior_logvar + (prior_var + (mu - prior_mu)**2) / var - 1)
        
        # Weighted sum of KL divergences for the mixture (PoE)
        if weights is not None:
            kl_loss = torch.sum(weights * kl_experts, dim=0)
        else:
            # If no weights are provided, use equal contribution from each expert
            kl_loss = torch.mean(kl_experts, dim=0)
        
        return kl_loss
        
    def calculate_loss(self, data):
        
        ret = self.model(data)
        
        kl_loss = self.kl_loss(mu=torch.cat(ret['mu_img'], ret['mu_csi'], 0),
                               logvar=torch.cat(ret['logvar_img'], ret['logvar_csi'], 0))
        recon_loss = self.recon_lossfunc(ret['re_img'], data['rimg']) / ret['re_img'].shape[0]
        
        loss = kl_loss * self.beta + recon_loss

        TEMP_LOSS = {
            'LOSS' : loss,
            'KL'   : kl_loss,
            'RECON': recon_loss
              }   
        
        PREDS = {
            'GT': data['rimg'],
            'PRED': ret['re_img'],
            'LAT': ret['z'],
            'TAG': data['tag'],
            'IND': data['ind']
        }
        
        return PREDS, TEMP_LOSS
        
    def plot_test(self, select_ind=None, select_num=8, autosave=False, **kwargs):
        figs: dict = {}
        self.losslog.generate_indices(select_ind, select_num)

        figs.update(self.losslog.plot_predict(plot_terms=('GT', 'PRED')))
        figs.update(self.losslog.plot_latent(plot_terms={'LAT'}))

        if autosave:
            for filename, fig in figs.items():
                fig.savefig(f"{self.save_path}{filename}")


if __name__ == "__main__":

    m = ThroughWallTrainer()
    print(ThroughWallTrainer.preprocess(ThroughWallTrainer, 1))
