from __future__ import print_function
import sys
from argparse import Namespace
import torch.nn.functional as F
import torch
from commonTraining import kl_divergence, kl_discrete, log_sum_exp, \
        print_in_epoch_summary, print_epoch_summary, mse_loss
from utils import logit
import numpy as np

SMALL = 1e-16

def nll_and_kl(recon_x, x, log_likelihood, a, b, logsample, z_discrete, logit_post, log_prior, mu, logvar, dataset_size, args=Namespace(), test=False):
    batch_size = x.size()[0]
    NLL = -1 * log_likelihood(recon_x, x)
    KL_zreal = -0.5 * (1. + logvar - mu**2 - logvar.exp())
    KL_beta = kl_divergence(a, b, prior_alpha=args.alpha0, log_beta_prior=np.log(1./args.alpha0), args=args).repeat(batch_size, 1) * (1. / dataset_size)

    # in test mode, our samples are essentially coming from a Bernoulli
    if not test:
        KL_discrete = kl_discrete(logit_post, logit(log_prior.exp()), logsample, args.temp, args.temp_prior)
    else:
        pi_prior = torch.exp(log_prior)
        pi_posterior = torch.sigmoid(logit_post)
        kl_1 = z_discrete * (pi_posterior + SMALL).log() + (1 - z_discrete) * (1 - pi_posterior + SMALL).log()
        kl_2 = z_discrete * (pi_prior + SMALL).log() + (1 - z_discrete) * (1 - pi_prior + SMALL).log()
        KL_discrete = kl_1 - kl_2

    return NLL, KL_zreal, KL_beta, KL_discrete

def cross_entropy(y, logits):
    return -torch.sum(y * torch.log(logits + SMALL), dim=1)

def elbo(recon_x, x, log_likelihood, a, b, logsample, z_discrete, logit_post, log_prior, mu, logvar, dataset_size, args=Namespace(), test=False):
    NLL, KL_zreal, KL_beta, KL_discrete = nll_and_kl(recon_x, x, log_likelihood, a, b, logsample, z_discrete, logit_post, log_prior, mu, logvar, dataset_size, args, test=test)
    return NLL.sum() + args.beta*(KL_zreal.sum() + KL_beta.sum() + KL_discrete.sum()), (NLL, KL_zreal, KL_beta, KL_discrete)

def train_MNIST(train_loader, model, log_likelihood, optimizer, epoch, args=Namespace()):
    model.train()
    model.double()
    train_loss = 0.

    for batch_idx, (data, zs) in enumerate(train_loader):

        data = torch.autograd.Variable(data.double())

        if args.cuda:
            data = data.cuda()

        recon_batch, logsample, logit_post, log_prior, mu, logvar, z_discrete, z_continuous = model(data)

        loss, (NLL, KL_zreal, KL_beta, KL_discrete) = elbo(recon_batch, data, log_likelihood,
                F.softplus(model.beta_a) + 0.01, (F.softplus(model.beta_b) + 0.01),
                logsample, z_discrete, logit_post, log_prior, mu, logvar, len(train_loader.dataset), args)

        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print_in_epoch_summary(epoch, batch_idx, len(data), len(train_loader.dataset), loss.data[0],
                                   NLL.sum().data[0],
                                   {'zreal': KL_zreal.sum().data[0], 'beta': KL_beta.sum().data[0],
                                    'discrete': KL_discrete.sum().data[0]})

    print_epoch_summary(epoch, train_loss / len(train_loader.dataset))
    return train_loss / len(train_loader.dataset)

def test_MNIST(test_loader, model, log_likelihood, epoch, args=Namespace()):
    model.eval()
    model.double()
    test_loss = 0

    for batch_idx, (data, zs) in enumerate(test_loader):
        data = torch.autograd.Variable(data.double())

        if args.cuda:
            data = data.cuda()

        recon_batch, logsample, logit_post, log_prior, mu, logvar, z_discrete, z_continuous =  model(data)
        loss, (NLL, KL_zreal, KL_beta, KL_discrete) = elbo(recon_batch, data, log_likelihood,
                F.softplus(model.beta_a) + 0.01, F.softplus(model.beta_b) + 0.01,
                logsample, z_discrete, logit_post, log_prior, mu, logvar, len(test_loader.dataset), args, test=True)

        test_loss += loss.data[0]

    test_loss /= len(test_loader.dataset)
    print('====> Test ELBO loss: {:.3f} '.format(test_loss))
    sys.stdout.flush()
    return test_loss