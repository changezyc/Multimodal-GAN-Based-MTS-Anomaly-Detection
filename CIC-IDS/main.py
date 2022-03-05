#!/usr/bin/env python
# coding: utf-8
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from minepy import MINE
import model
import anomaly_detection
import utils
import time
import pickle
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(filename='train.log', level=logging.DEBUG)


class SeqDataset(Dataset):

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


class CosineDataset(Dataset):

    def __init__(self, file_path):
        with open(file_path, "rb") as f:
            self.data_list = pickle.load(f)

    def __getitem__(self, item):

        return np.expand_dims(self.data_list[item][0], axis=0), self.data_list[item][1]

    def __len__(self):
        return len(self.data_list)


def discriminator_iteration(inputs_c, alpha):
    optim_discriminator.zero_grad()
    x_c = inputs_c.float()
    valid_x_score = discriminator(x_c)
    critic_score_valid_x = torch.squeeze(valid_x_score)
    z = torch.randn(batch_size, 40).to(device)
    # z = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 40))
    # z = torch.from_numpy(z).to(torch.float32).to(device)
    fake_x_r = generator_rnn(z)
    fake_x_r_c = torch.empty(batch_size, 77, 77).to(device)
    for i in range(len(fake_x_r)):
        fake_x_r_c[i] = utils.calculate_Cosine_matirx_intensor(fake_x_r[i])
    fake_x_r_c = fake_x_r_c.unsqueeze(1)
    fake_x_c = generator_cnn(z)
    fake_x = alpha * fake_x_c + (1 - alpha) * fake_x_r_c
    fake_x_score = discriminator(fake_x)
    critic_score_fake_x = torch.squeeze(fake_x_score)
    wl = -critic_score_valid_x + critic_score_fake_x
    wl = wl.mean()
    gp_loss = utils.compute_gradient_penalty(discriminator, x_c, fake_x)
    lambda_gp = 10
    loss = wl + lambda_gp * gp_loss
    loss.backward()
    optim_discriminator.step()
    return loss


def generator_cnn_iteration():
    optim_generator_cnn.zero_grad()
    z = torch.randn(batch_size, 40).to(device)
    # z = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 40))
    # z = torch.from_numpy(z).to(torch.float32).to(device)
    x_ = generator_cnn(z)
    fake_x_score = discriminator(x_)
    loss_gen = -torch.mean(fake_x_score)
    loss_gen.backward()
    optim_generator_cnn.step()
    return loss_gen


def generator_rnn_iteration():
    optim_generator_rnn.zero_grad()
    z = torch.randn(batch_size, 40).to(device)
    # z = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 40))
    # z = torch.from_numpy(z).to(torch.float32).to(device)
    x_ = generator_rnn(z)
    new_x_ = torch.empty(batch_size, 77, 77).to(device)
    for i in range(len(x_)):
        new_x_[i] = utils.calculate_Cosine_matirx_intensor(x_[i])
    new_x_ = new_x_.unsqueeze(1)
    fake_x_score = discriminator(new_x_)
    loss_gen = -torch.mean(fake_x_score)
    loss_gen.backward()
    optim_generator_cnn.step()
    return loss_gen


def encoder_rnn_iteration(inputs, generator):
    x = inputs.float()
    optim_encoder_rnn.zero_grad()
    zr = torch.randn(batch_size, 100, 40).to(device)
    # z = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 40))
    # z = torch.from_numpy(z).to(torch.float32).to(device)
    loss_enc = mse_loss(generator(encoder_rnn(x)), x)
    loss_enc.backward()
    optim_encoder_rnn.step()
    return loss_enc


def encoder_cnn_iteration(inputs, generator):
    x = inputs.float()
    optim_encoder_cnn.zero_grad()
    zr = torch.randn(batch_size, 100, 40).to(device)
    # z = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 40))
    # z = torch.from_numpy(z).to(torch.float32).to(device)
    x_r_c = torch.empty(batch_size, 77, 77).to(device)
    for i in range(len(inputs)):
        x_r_c[i] = utils.calculate_Cosine_matirx_intensor(inputs[i])
    x_r_c = x_r_c.unsqueeze(1)
    loss_enc = mse_loss(generator(encoder_cnn(x_r_c)), x_r_c)
    loss_enc.backward()
    optim_encoder_cnn.step()
    return loss_enc


def train(n_epochs):
    logging.debug('Starting training')
    discriminator_epoch_loss = list()
    generator_cnn_epoch_loss = list()
    generator_rnn_epoch_loss = list()
    for epoch in range(n_epochs):
        print('======================================================')
        start = time.time()
        logging.debug('Epoch {}'.format(epoch))
        print('Epoch {}'.format(epoch))
        n_discriminators = 5
        n_gens = 1
        discriminator_nd_loss = list()
        generator_cnn_ng_loss = list()
        generator_rnn_ng_loss = list()

        for i in range(n_discriminators):
            discriminator_loss = list()
            for batch, (inputs_c, labels) in enumerate(train_loader_cos):
                inputs_c = inputs_c.to(device)
                loss_discriminator = discriminator_iteration(inputs_c, alpha=1).item()
                discriminator_loss.append(loss_discriminator)
            discriminator_nd_loss.append(torch.mean(torch.tensor(discriminator_loss)))

        for i in range(n_gens):
            generator_loss_cnn = list()
            generator_loss_rnn = list()
            for batch, _ in enumerate(train_loader_cos):
                loss_generator_cnn = generator_cnn_iteration().item()
                loss_generator_rnn = generator_rnn_iteration().item()
                generator_loss_cnn.append(loss_generator_cnn)
                generator_loss_rnn.append(loss_generator_rnn)
            generator_cnn_ng_loss.append(torch.mean(torch.tensor(generator_loss_cnn)))
            generator_rnn_ng_loss.append(torch.mean(torch.tensor(generator_loss_rnn)))

        discriminator_epoch_loss.append((torch.mean(torch.tensor(discriminator_nd_loss))).item())
        generator_cnn_epoch_loss.append((torch.mean(torch.tensor(generator_cnn_ng_loss))).item())
        generator_rnn_epoch_loss.append((torch.mean(torch.tensor(generator_rnn_ng_loss))).item())
        logging.debug('Generator,Discriminator training done in epoch {}'.format(epoch))
        logging.debug('\ndiscriminator loss {:.3f}\ngenerator cnn loss {:.3f}\ngenerator rnn loss {:.3f}\n'.format(discriminator_epoch_loss[-1], generator_cnn_epoch_loss[-1], generator_rnn_epoch_loss[-1]))
        print('Generator,Discriminator training done in epoch {}'.format(epoch))
        print('\ndiscriminator loss {:.3f}\ngenerator cnn loss {:.3f}\ngenerator rnn loss {:.3f}\n'.format(discriminator_epoch_loss[-1], generator_cnn_epoch_loss[-1], generator_rnn_epoch_loss[-1]))
        end = time.time()
        print('time used: %.2f seconds' % (end - start))
        print('======================================================')
        if (epoch + 1) % 10 == 0:
            torch.save(generator_cnn.state_dict(), generator_cnn_path)
            torch.save(generator_rnn.state_dict(), generator_rnn_path)
            torch.save(discriminator.state_dict(), discriminator_path)
            torch.save(optim_generator_cnn.state_dict(), optim_generator_cnn_path)
            torch.save(optim_generator_rnn.state_dict(), optim_generator_rnn_path)
            torch.save(optim_discriminator.state_dict(), optim_discriminator_path)


def train_encoder_rnn(n_encoder_epochs, generator_rnn):
    encoder_epoch_loss = list()
    for epoch in range(n_encoder_epochs):
        start = time.time()
        loss_list = list()
        # z = torch.randn(batch_size, 40).to(device)
        for batch, (inputs, labels) in enumerate(train_loader_seq):
            inputs = inputs.to(device)
            loss = encoder_rnn_iteration(inputs, generator_rnn)
            loss_list.append(loss)
        encoder_epoch_loss.append((torch.mean(torch.tensor(loss_list))).item())
        print('Epoch {}'.format(epoch + 1))
        print('\nencoder loss {:.3f}\n'.format(encoder_epoch_loss[-1]))
        end = time.time()
        print('time used: %.2f seconds' % (end - start))
        print('======================================================')
        if (epoch + 1) % 50 == 0:
            torch.save(encoder_rnn.state_dict(), encoder_rnn_path)
            torch.save(optim_encoder_rnn.state_dict(), optim_encoder_rnn_path)


def train_encoder_cnn(n_encoder_epochs, generator_cnn):
    encoder_epoch_loss = list()
    for epoch in range(n_encoder_epochs):
        start = time.time()
        loss_list = list()
        # z = torch.randn(batch_size, 40).to(device)
        for batch, (inputs, labels) in enumerate(train_loader_seq):
            inputs = inputs.to(device)
            loss = encoder_cnn_iteration(inputs, generator_cnn)
            loss_list.append(loss)
        encoder_epoch_loss.append((torch.mean(torch.tensor(loss_list))).item())
        print('Epoch {}'.format(epoch + 1))
        print('\nencoder loss {:.3f}\n'.format(encoder_epoch_loss[-1]))
        end = time.time()
        print('time used: %.2f seconds' % (end - start))
        print('======================================================')
        if (epoch + 1) % 50 == 0:
            torch.save(encoder_cnn.state_dict(), encoder_cnn_path)
            torch.save(optim_encoder_cnn.state_dict(), optim_encoder_cnn_path)


if __name__ == "__main__":
    mse_loss = torch.nn.MSELoss()
    fPath_Attack_v0 = 'cicids_FridayAfternoon1_test.csv'
    fPath_Normal_v0 = 'cicids_Monday_train.csv'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_train, label_train, dataset_test, label_test = utils.process_datacsv_to_normal_distribution(fPath_Normal_v0, fPath_Attack_v0)
    train_dataset_seq = SeqDataset(dataset_train, label_train, 100, 50)
    test_dataset_seq = SeqDataset(dataset_test, label_test, 100, 50)
    train_dataset_cos = CosineDataset('cosine_train')
    test_dataset_cos = CosineDataset('cosine_test')
    batch_size = 64
    train_loader_seq = DataLoader(train_dataset_seq, batch_size=batch_size, drop_last=True)
    test_loader_seq = DataLoader(test_dataset_seq, batch_size=batch_size, drop_last=True)
    train_loader_cos = DataLoader(train_dataset_cos, batch_size=batch_size, drop_last=True)
    test_loader_cos = DataLoader(test_dataset_cos, batch_size=batch_size, drop_last=True)
    lr = 1e-6
    # lr = 1e-6
    features_shape = 77
    generator_cnn_path = 'models/generator_cnn.pt'
    generator_rnn_path = 'models/generator_rnn.pt'
    discriminator_path = 'models/discriminator.pt'
    encoder_rnn_path = 'models/encoder_rnn.pt'
    encoder_cnn_path = 'models/encoder_cnn.pt'
    optim_generator_cnn_path = 'models/optim_generator_cnn.pt'
    optim_generator_rnn_path = 'models/optim_generator_rnn.pt'
    optim_discriminator_path = 'models/optim_discriminator.pt'
    optim_encoder_rnn_path = 'models/optim_encoder_rnn.pt'
    optim_encoder_cnn_path = 'models/optim_encoder_cnn.pt'
    generator_cnn = model.Generator_C()
    generator_rnn = model.Generator_R()
    discriminator = model.Discriminator()
    encoder_rnn = model.Encoder_R()
    encoder_cnn = model.Encoder_C()

    generator_cnn.load_state_dict(torch.load(generator_cnn_path))
    generator_rnn.load_state_dict(torch.load(generator_rnn_path))
    discriminator.load_state_dict(torch.load(discriminator_path))
    encoder_rnn.load_state_dict(torch.load(encoder_rnn_path))
    encoder_cnn.load_state_dict(torch.load(encoder_cnn_path))

    generator_cnn = generator_cnn.to(device)
    generator_rnn = generator_rnn.to(device)
    discriminator = discriminator.to(device)
    encoder_rnn = encoder_rnn.to(device)
    encoder_cnn = encoder_cnn.to(device)

    optim_generator_cnn = optim.Adam(generator_cnn.parameters(), lr=lr, betas=(0, 0.9))
    optim_generator_rnn = optim.Adam(generator_rnn.parameters(), lr=lr, betas=(0, 0.9))
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=lr, betas=(0, 0.9))
    optim_encoder_rnn = optim.Adam(encoder_rnn.parameters(), lr=0.00001, betas=(0.9, 0.999))
    optim_encoder_cnn = optim.Adam(encoder_cnn.parameters(), lr=0.00001, betas=(0.9, 0.999))
    # optim_generator_cnn.load_state_dict(torch.load(optim_generator_cnn_path))
    # optim_generator_rnn.load_state_dict(torch.load(optim_generator_rnn_path))
    # optim_discriminator.load_state_dict(torch.load(optim_discriminator_path))
    # optim_encoder_rnn.load_state_dict(torch.load(optim_encoder_rnn_path))
    # optim_encoder_cnn.load_state_dict(torch.load(optim_encoder_cnn_path))

    # train(n_epochs=100)

    # train_encoder_rnn(n_encoder_epochs=200, generator_rnn=generator_rnn)
    # train_encoder_cnn(n_encoder_epochs=200, generator_cnn=generator_cnn)

    anomaly_detection.test(test_loader_seq, generator_rnn, generator_cnn, discriminator, encoder_rnn, encoder_cnn)
