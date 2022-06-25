import torch
import torch.nn as nn
import torch.distributions as dist
from torch import nn
from torch.nn import functional as F

import numpy as np
import math
import random

from maths import sigmoid
from utils import impute_observed_data


"""
Credits: 
ELBO, IW-ELBO, and DReG IW-ELBO implementations follow: https://github.com/iffsid/DReG-PyTorch
"""


def check_valid_estimate(model):
    for param in model.parameters():
        if torch.isnan(param).any():
            raise ValueError('Nan exists in model parameter values')


class MirtIWVAE(nn.Module):

    def __init__(
        self,
        input_size, 
        hidden_size, 
        depth,
        latent_size, 
        pl,
        dropout_rate=None,
        activation_fn=nn.ReLU(inplace=True),
        init_a=None,
        init_b=None,
        init_c=None,
        init_d=None,
        restrict_a=None,
        learn_sigma=True,
        seed=1,
        ):
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        #  Encoder
        encoder = [
            nn.Linear(input_size, hidden_size),
            activation_fn,
            ]
        if dropout_rate:
            encoder.append(
                nn.Dropout(dropout_rate)
                )
        for _ in range(depth):
            encoder.extend([
                nn.Linear(hidden_size, hidden_size),
                activation_fn
                ])
            if dropout_rate:
                encoder.append(
                    nn.Dropout(dropout_rate)
                    )
        self.encoder = nn.Sequential(
            *encoder
            )

        self.mu_x = nn.Linear(
            hidden_size, latent_size
            )
        self.log_var_x = nn.Linear(
            hidden_size, latent_size
            )

        # MIRT Parameters
        self.pl = pl
        self.J = input_size
        self.K = latent_size
        self.restrict_a = restrict_a
        self.unrestricted_param = {
            'a': None, 'b': None, 'c': None, 'd': None
            }

        self.init_a(init_a)
        self.init_b(init_b)
        self.init_c(init_c)
        self.init_d(init_d)

        self.learn_sigma = learn_sigma
        self.fit_time = None 
        
    def init_a(self, init_value=None):
        init_value = init_value or torch.randn(
            self.J, self.K) * 0.1

        self.unrestricted_param['a'] = nn.Parameter(
            init_value, requires_grad=False
            )
        self.register_parameter(
            'unrestrict_a', self.unrestricted_param['a']
            )

    def init_b(self, init_value=None):
        init_value = init_value or torch.randn(
            self.J) * 0.1
        self.unrestricted_param['b'] = nn.Parameter(
            init_value, requires_grad=False
            )
        self.register_parameter(
            'unrestrict_b', self.unrestricted_param['b']
            )

    def init_c(self, init_value=None):
        if self.pl > 2:
            init_value = init_value or torch.randn(
                self.J) * 0.1 - 4
            self.unrestricted_param['c'] = nn.Parameter(
                init_value, requires_grad=False
                )
            self.register_parameter(
                'unrestrict_c', self.unrestricted_param['c']
                )

    def init_d(self, init_value=None):
        if self.pl > 3:
            init_value = init_value or torch.randn(
                self.J) * 0.1 + 4
            self.unrestricted_param['d'] = nn.Parameter(
                init_value, requires_grad=False
                )
            self.register_parameter(
                'unrestrict_d', self.unrestricted_param['d']
                )

    @property
    def a(self):
        a = self.unrestricted_param['a']
        if self.restrict_a:
            a = self.restrict_a(a)
        return a

    @property
    def b(self):
        return self.unrestricted_param['b']

    @property
    def c(self):
        c = self.unrestricted_param['c']
        if c != None:
            c = torch.sigmoid(c)
        else:
            c = torch.zeros(
                self.J, requires_grad=False
                )
        return c

    @property
    def d(self):
        d = self.unrestricted_param['d']
        if d != None:
            d = torch.sigmoid(d)
        else:
            d = torch.ones(
                self.J, requires_grad=False
                )
        return d

    def get_param(self, param):
        if param == 'a':
            return self.a
        elif param == 'b':
            return self.b
        elif param == 'c':
            return self.c
        elif param == 'd':
            return self.d
        else:
            raise ValueError(f'Unrecognized param: {param}')

    def allow_update_decoder(self, *params):
        for param in params:
            if self.unrestricted_param[param] != None:
                self.unrestricted_param[param].requires_grad = True

    def forward(self, y, R=1, S=1):
        y = y.float()
        bsz = y.shape[0]

        _encode = self.encoder(y)
        mu_x = self.mu_x(_encode)
        if self.learn_sigma:
            sigma_x = torch.exp(
                0.5 * self.log_var_x(_encode)
                )
        else:
            sigma_x = torch.ones_like(
                mu_x, requires_grad=False
                )
        qx_y = dist.Normal(mu_x, sigma_x)

        x = qx_y.rsample(
            torch.Size([S, R])
            )
        y_prob = self.c + (self.d - self.c) * torch.sigmoid(
            x @ self.a.T + self.b
            )
        py_x = dist.Bernoulli(probs=y_prob)

        return py_x, qx_y, x

def elbo(y, y_observed, x, py_x, qx_y, kl_weight=1.):
    y = y.float()
    S, R, B, K = x.shape
    J = y.shape[1]

    if R != 1:
        raise ValueError('Cannot compute ELBO when multiple MC samples are drawn.')

    log_py_x = py_x.log_prob(y) * y_observed
    log_py_x = log_py_x.sum(-1)
    px = dist.Normal(
        torch.zeros_like(x), torch.ones_like(x)
        )
    kl = dist.kl_divergence(qx_y, px).sum(-1)

    return torch.mean(log_py_x - kl * kl_weight)


def iw_elbo(y, y_observed, x, py_x, qx_y):

    y = y.float()
    S, R, B, K = x.shape
    J = y.shape[1]

    log_py_x = py_x.log_prob(y) * y_observed
    log_py_x = log_py_x.sum(-1)
    log_px = dist.Normal(
        torch.zeros_like(x),
        torch.ones_like(x)
        ).log_prob(x).sum(-1)
    log_qx_y = qx_y.log_prob(x).sum(-1)
    log_w = log_py_x + log_px - log_qx_y

    return torch.mean(torch.logsumexp(log_w, 1) - math.log(R))


def iw_elbo_dreg(y, y_observed, x, py_x, qx_y):

    y = y.float()
    S, R, B, K = x.shape
    J = y.shape[1]

    log_py_x = py_x.log_prob(y) * y_observed
    log_py_x = log_py_x.sum(-1)
    log_px = dist.Normal(
        torch.zeros_like(x),
        torch.ones_like(x)
        ).log_prob(x).sum(-1)

    qx_y_detached = qx_y.__class__(
        qx_y.loc.detach(), qx_y.scale.detach()
        )
    log_qx_y = qx_y_detached.log_prob(x).sum(-1)
    log_w = log_py_x + log_px - log_qx_y

    with torch.no_grad():
        reweight = torch.exp(
            log_w - torch.logsumexp(log_w, 1, keepdim=True)
            )
        x.register_hook(
            lambda grad: grad * torch.pow(
                reweight.unsqueeze(-1), 2)
            )

    return torch.mean((reweight * log_w).sum(1))


def predict(model, data_loader):
    with torch.no_grad():
        model.eval()
        y_pred = []
        y_true = []
        learn_x_mu = []
        for y_obs in data_loader:
            observed, y = impute_observed_data(y_obs)

            py_x, qx_y, x = model(y)
            _pred_prob = py_x.probs.mean((0, 1)).cpu().data.numpy()
            y_recon = (_pred_prob >= 0.5) * 1

            y_pred.append(y_recon)
            y_true.append(y_obs.cpu().data.numpy())
            learn_x_mu.append(qx_y.loc.cpu().data.numpy())

        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        learn_x_mu = np.concatenate(learn_x_mu, axis=0)

        pred_a = model.a.cpu().data.numpy()
        pred_b = model.b.cpu().data.numpy()
        pred_c = model.c.cpu().data.numpy()
        pred_d = model.d.cpu().data.numpy()

        pred_prob = pred_c + (pred_d - pred_c) * sigmoid(
            learn_x_mu @ pred_a.T + pred_b)

    return y_true, y_pred, pred_prob