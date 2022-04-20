#!/usr/bin/env python
# coding: utf-8
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import model
import anomaly_detection
import utils
import time
from torch.autograd import Variable
from anomaly_detection import *
import warnings
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


def discriminator_iteration(inputs):
    optim_discriminator.zero_grad()
    x = inputs.float()
    valid_x_score = discriminator(x)
    valid = Variable(torch.cuda.FloatTensor(np.ones((x.size(0), 1))), requires_grad=False)
    fake = Variable(torch.cuda.FloatTensor(np.zeros((x.size(0), 1))), requires_grad=False)
    loss_x_real = bce_loss(valid_x_score, valid)
    # z = torch.randn(batch_size, 40).to(device)
    # z = torch.rand(batch_size, 40).to(device)
    z = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 40))
    z = torch.from_numpy(z).to(torch.float32).to(device)
    x_ = generator(z)
    fake_x_score = discriminator(x_)
    loss_x_fake = bce_loss(fake_x_score, fake)
    loss = loss_x_real + loss_x_fake
    loss.backward()
    optim_discriminator.step()
    return loss


def generator_iteration(inputs):
    optim_generator.zero_grad()
    x = inputs.float()
    z = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 40))
    z = torch.from_numpy(z).to(torch.float32).to(device)
    valid = Variable(torch.cuda.FloatTensor(np.ones((x.size(0), 1))), requires_grad=False)
    fake_x_score = discriminator(generator(z))
    loss = bce_loss(fake_x_score, valid)
    loss.backward()
    optim_generator.step()
    return loss


def train(n_epochs):
    logging.debug('Starting training')
    discriminator_epoch_loss = list()
    generator_epoch_loss = list()
    for epoch in range(n_epochs):
        print('======================================================')
        start = time.time()
        logging.debug('Epoch {}'.format(epoch))
        print('Epoch {}'.format(epoch))
        n_discriminators = 2
        n_generators = 7
        discriminator_nd_loss = list()
        generator_ng_loss = list()
        for i in range(n_discriminators):
            discriminator_loss = list()
            for batch, (inputs, _) in enumerate(train_loader):
                inputs = inputs.to(device)
                loss_dis = discriminator_iteration(inputs)
                discriminator_loss.append(loss_dis)
            discriminator_nd_loss.append(torch.mean(torch.tensor(discriminator_loss)))
        for i in range(n_generators):
            generator_loss = list()
            for batch, (inputs, _) in enumerate(train_loader):
                inputs = inputs.to(device)
                loss_gen = generator_iteration(inputs)
                generator_loss.append(loss_gen)
            generator_ng_loss.append(torch.mean(torch.tensor(generator_loss)))
        discriminator_epoch_loss.append(torch.mean(torch.tensor(discriminator_nd_loss)))
        generator_epoch_loss.append(torch.mean(torch.tensor(generator_ng_loss)))
        logging.debug('GAN training done in epoch {}'.format(epoch))
        logging.debug('\ndiscriminator loss {:.3f}\ngenerator loss {:.3f} \n'.format(discriminator_epoch_loss[-1], generator_epoch_loss[-1]))
        print('GAN training done in epoch {}'.format(epoch))
        print('discriminator loss: {:.3f}\ngenerator loss: {:.3f}\n'.format(discriminator_epoch_loss[-1], generator_epoch_loss[-1]))
        end = time.time()
        print('time used: %.2f seconds' % (end - start))
        print('======================================================')
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), generator.generator_path)
            torch.save(discriminator.state_dict(), discriminator.discriminator_path)
            torch.save(optim_generator.state_dict(), optim_generator_path)
            torch.save(optim_discriminator.state_dict(), optim_discriminator_path)


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fPath_Normal_v0 = 'cicids_Monday_train.csv'
    fPath_Attack_v0 = 'cicids_FridayAfternoon1_test.csv'

    dataset_train, label_train, dataset_test, label_test = utils.process_data_csv(fPath_Normal_v0, fPath_Attack_v0)

    batch_size = 64
    train_dataset = SignalDataset(dataset_train, label_train, 100, 50)
    test_dataset = SignalDataset(dataset_test, label_test, 100, 50)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)
    
    lr = 1e-6

    features_shape = 77
    generator_path = 'models/generator.pt'
    discriminator_path = 'models/discriminator.pt'

    optim_generator_path = 'models/optim_generator.pt'
    optim_discriminator_path = 'models/optim_discriminator.pt.pt'

    generator = model.Generator(generator_path, features_shape)
    discriminator = model.Discriminator(discriminator_path, features_shape)

    generator.load_state_dict(torch.load(generator_path))
    discriminator.load_state_dict(torch.load(discriminator_path))

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCELoss()

    optim_generator = optim.RMSprop(generator.parameters(), lr=lr)
    optim_discriminator = optim.RMSprop(discriminator.parameters(), lr=lr)

    # optim_generator.load_state_dict(torch.load(optim_generator_path))
    # optim_discriminator.load_state_dict(torch.load(optim_discriminator_path))

    # train(n_epochs=100)

    test(test_loader, generator, discriminator)
