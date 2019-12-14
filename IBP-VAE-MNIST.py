import sys
import os
import numpy as np
import time
import argparse

import torch.nn.init
import torch.optim as optim
from models import S_IBP_Concrete_MNIST
import training

parser = argparse.ArgumentParser(description='VAEs for the Indian Buffet Process (IBP)')

parser.add_argument('--dataset', type=str, default='Simulated',
                    help='dataset to train on')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--D', type=int, default=64*64, metavar='N',
                    help='dimension of simulated signal')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log-epoch', type=int, default=1, metavar='N',
                    help='wait every epochs')
parser.add_argument('--train-from', type=str, default=None, metavar='M',
                    help='model to train from, if any')
parser.add_argument('--load-data', type=str, default=None,
                    help='load dataset')
parser.add_argument('--savefile', type=str, default='condVT',
                    help='testsave')
parser.add_argument('--truncation', type=int, default=10,
                    help='number of sticks')
parser.add_argument('--alpha0', type=float, default=10.,
                    help='prior alpha for stick breaking Betas')
parser.add_argument('--repeat-v', type=int, default=1,
                    help='number of v samples to take (to reduce variance on KL)')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--hidden', type=int, default=500, help='hidden states')

parser.add_argument('--iwae', type=bool, default=False, help='use IWAE instead of elbo on test')
parser.add_argument('--n-samples', type=int, default=32, help='number of samples for calculating IWAE loss')
# BBVI specific
parser.add_argument('--no-cv', action='store_true', default=False,
                    help='control variates')
parser.add_argument('--n-cv-samples', type=int, default=3, help='number of samples for calculating control variates')
# concrete specific
parser.add_argument('--temp', type=float, default=1.,
                    help='temperature for concrete')
parser.add_argument('--temp_prior', type=float, default=0.5,
                    help='temperature for concrete prior')
parser.add_argument('--mode', type=float, default=1,
                    help='main program mode')
parser.add_argument('--checkpt', type=str, default=None,
                    help='checkpoint path')
parser.add_argument('--beta', type=float, default=1.0,
                    help='beta to increase KL weight')

# determinisim
np.random.seed(0)
torch.manual_seed(0)

global args
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.cuda:
    newTensor = torch.cuda.DoubleTensor
else:
    newTensor = torch.DoubleTensor

#parameters
SMALL = 1e-16


def log_likelihood(pred, data):
    return data.view(-1, 784) * (pred + SMALL).log() + (1 - data.view(-1, 784)) * (1 - pred + SMALL).log()


model_kwargs = {
    'dataset': args.dataset,
    'max_truncation_level': args.truncation,
    'alpha0': args.alpha0,
}
eval_kwargs = {
    'log_likelihood': log_likelihood,
    'args': args,
}

weight = 1

model_cls = S_IBP_Concrete_MNIST
model_kwargs['temp'] = args.temp
model_kwargs['hidden'] = args.hidden
trainer = training.train_MNIST
validator = training.test_MNIST

model = model_cls(**model_kwargs)
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
eval_kwargs['model'] = model
if not os.path.isdir('models'):
    os.mkdir('models')

train_scores = np.zeros(args.epochs)
validation_scores = np.zeros(args.epochs)
test_scores = np.zeros(args.epochs)
epoch_times = np.zeros(args.epochs)

best_valid = 10000

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

import torchvision.datasets as datasets
import torchvision.transforms as transforms

mnist = datasets.MNIST('data/MNIST', train=True, download=True,
                       transform=transforms.ToTensor())
mnistTest = datasets.MNIST('data/MNIST', train=False, download=True,
                           transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(mnist,
                                          batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(mnistTest,
                                         batch_size=args.batch_size, shuffle=False, **kwargs)

epochStart = 1

if args.mode==1 and args.train_from is not None:
    print("*******Mode:resume checkpoint")
    if os.path.isfile(args.train_from):
        print("=> loading checkpoint '{}'".format(args.train_from))
        checkpoint = torch.load(args.train_from)
        model.load_state_dict(checkpoint)
        epochStart = 35 + 1
    else:
        print("=> no checkpoint found at '{}'".format(args.train_from))
        sys.exit(1)

if args.mode==1:
    print("*******Mode: training")
    start = time.time()

    for epoch in range(epochStart, args.epochs + epochStart):
        train_scores[epoch - 1] = trainer(train_loader, model, log_likelihood, optimizer, epoch, args)

        if epoch % 5 == 0:
            torch.save(model.state_dict(), 'models/beta'+str(args.beta)+'_epoch_{}.pt'.format(epoch))
