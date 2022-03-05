#!/usr/bin/env python
# coding: utf-8
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import model
from torch.autograd import Variable
from anomaly_detection import *
from utils import *
import warnings
import time

warnings.filterwarnings("ignore")

logging.basicConfig(filename='train.log', level=logging.DEBUG)


class SignalDataset(Dataset):

    def __init__(self, data_numpy, lable_numpy, seq_len, stride):
        self.data_numpy = data_numpy
        self.lable_numpy = lable_numpy
        self.seq_len = seq_len
        self.stride = stride

    def __getitem__(self, item):
        s = slice(item * self.stride, item * self.stride + self.seq_len)
        return self.data_numpy[s], self.lable_numpy[s].max()

    def __len__(self):
        return (len(self.data_numpy) - self.seq_len) // self.stride + 1


def autoencoder_iteration(inputs):
    optim_autoencoder.zero_grad()
    x = inputs.float()
    x_ = autoencoder(x)
    loss = mse_loss(x, x_)
    loss.backward()
    optim_autoencoder.step()
    return loss


def train(n_epochs):
    logging.debug('Starting training')
    autoencoder_epoch_loss = list()
    for epoch in range(n_epochs):
        print('======================================================')
        start = time.time()
        logging.debug('Epoch {}'.format(epoch))
        print('Epoch {}'.format(epoch))
        autoencoder_loss = list()
        for batch, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            loss_dis = autoencoder_iteration(inputs)
            autoencoder_loss.append(loss_dis)
        autoencoder_epoch_loss.append(torch.mean(torch.tensor(autoencoder_loss)))
        logging.debug('autoencoder training done in epoch {}'.format(epoch))
        logging.debug('\nautoencoder loss {:.3f}\n'.format(autoencoder_epoch_loss[-1]))
        print('autoencoder training done in epoch {}'.format(epoch))
        print('autoencoder loss: {:.3f}\n'.format(autoencoder_epoch_loss[-1]))
        end = time.time()
        print('time used: %.2f seconds' % (end - start))
        print('======================================================')
        if (epoch + 1) % 10 == 0:
            torch.save(autoencoder.state_dict(), autoencoder_path)
            torch.save(optim_autoencoder.state_dict(), optim_autoencoder_path)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fPath_Normal = 'SWaT_train.csv'
    fPath_Attack = 'SWaT_test.csv'
    dataset_train, label_train, dataset_test, label_test = process_data_csv(fPath_Normal, fPath_Attack)
    batch_size = 64
    mse_loss = torch.nn.MSELoss()
    train_dataset = SignalDataset(dataset_train, label_train, 100, 50)
    test_dataset = SignalDataset(dataset_test, label_test, 100, 50)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    lr = 1e-4

    features_shape = 51
    autoencoder_path = 'models/autoencoder.pt'
    optim_autoencoder_path = 'models/optim_autoencoder.pt'

    autoencoder = model.LSTMAutoEncoder(autoencoder_path, features_shape)

    autoencoder.load_state_dict(torch.load(autoencoder_path))

    autoencoder = autoencoder.to(device)

    optim_autoencoder = optim.RMSprop(autoencoder.parameters(), lr=lr)

    # optim_autoencoder.load_state_dict(torch.load(optim_autoencoder_path))

    # train(n_epochs=100)

    test(test_loader, autoencoder)