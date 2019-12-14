import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from utils import logit
from commonModels import init_weights, reparametrize, reparametrize_discrete, reparametrize_gaussian

SMALL = 1e-16

class S_IBP_Concrete_MNIST(nn.Module):
    def __init__(self, max_truncation_level=50, temp=1., alpha0=10., dataset='MNIST', hidden=500, X_dim = 28*28):
        super(S_IBP_Concrete_MNIST, self).__init__()
        self.temp = temp
        self.dataset = dataset
        self.truncation = max_truncation_level
        self.z_dim=max_truncation_level

        self.fc1_encode = nn.Linear(X_dim, hidden)
        self.fc2_encode = nn.Linear(hidden, self.truncation * 3)


        # generate: deep
        self.fc1_decode = nn.Linear(self.truncation, hidden)
        self.fc2_decode = nn.Linear(hidden, X_dim)

        a_val = np.log(np.exp(alpha0) - 1) # inverse softplus
        b_val = np.log(np.exp(1.) - 1)
        self.beta_a = nn.Parameter(torch.Tensor(self.truncation).zero_() + a_val)
        self.beta_b = nn.Parameter(torch.Tensor(self.truncation).zero_() + b_val)

        init_weights([self.fc1_encode, self.fc2_encode, self.fc1_decode, self.fc2_decode])

    def encode(self, x):
        x = x.view(-1, 784)
        h3 = F.relu(self.fc1_encode(x))

        logit_x, mu, logvar = torch.split(self.fc2_encode(h3),
                                          self.truncation, 1)

        return logit_x, mu, logvar

    def decode(self, z):
        x = F.sigmoid(
            self.fc2_decode(
                F.relu(
                    self.fc1_decode(z)
                )
            )
        )
        return x

    def forward(self, x, log_prior=None):
        batch_size = x.size(0)
        truncation = self.beta_a.size(0)
        beta_a = F.softplus(self.beta_a) + 0.01
        beta_b = F.softplus(self.beta_b) + 0.01

        # might be passed in for IWAE
        if log_prior is None:
            log_prior = reparametrize(
                beta_a.view(1, truncation).expand(batch_size, truncation),
                beta_b.view(1, truncation).expand(batch_size, truncation),
                ibp=True, log=True)

        logit_x, mu, logvar = self.encode(x)
        logit_post = logit_x + logit(log_prior.exp())

        logsample = reparametrize_discrete(logit_post, self.temp)
        z_discrete = F.sigmoid(logsample) # binary
        z_continuous = reparametrize_gaussian(mu, logvar)

        # zero-temperature rounding
        if not self.training:
            z_discrete = torch.round(z_discrete)

        inputForDecoder = z_discrete * z_continuous

        dec = self.decode(inputForDecoder)

        return dec, logsample, logit_post, log_prior, mu, logvar, z_discrete, z_continuous