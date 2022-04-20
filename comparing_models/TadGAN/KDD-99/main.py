#!/usr/bin/env python
# coding: utf-8

import logging
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import model
import anomaly_detection
from utils import *
import time

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


def critic_x_iteration(inputs):
    optim_cx.zero_grad()
    x = inputs.transpose(0, 1).to(torch.float32)
    valid_x_score = critic_x(x)
    valid_x_score = torch.squeeze(valid_x_score)
    critic_score_valid_x = torch.mean(valid_x_score)
    z = torch.empty(100, batch_size, latent_space_dim).uniform_(0, 1).to(device)
    x_ = decoder(z)
    fake_x_score = critic_x(x_)
    fake_x_score = torch.squeeze(fake_x_score)
    critic_score_fake_x = torch.mean(fake_x_score)
    wl = critic_score_fake_x - critic_score_valid_x
    gp_loss = compute_gradient_penalty(critic_x, x, x_)
    lambda_gp = 10
    loss = wl + lambda_gp * gp_loss
    loss.backward()
    optim_cx.step()

    return loss


def critic_z_iteration(inputs):
    optim_cz.zero_grad()
    x = inputs.transpose(0, 1).to(torch.float32)
    z = encoder(x)
    fake_z_score = critic_z(z)
    fake_z_score = torch.squeeze(fake_z_score)
    critic_score_fake_z = torch.mean(fake_z_score)
    z_ = torch.empty(100, batch_size, latent_space_dim).uniform_(0, 1).to(device)
    valid_z_score = critic_z(z_)
    valid_z_score = torch.squeeze(valid_z_score)
    critic_score_valid_z = torch.mean(valid_z_score)
    wl = critic_score_fake_z - critic_score_valid_z
    gp_loss = compute_gradient_penalty(critic_z, z, z_)
    lambda_gp = 10
    loss = wl + lambda_gp * gp_loss
    loss.backward()
    optim_cz.step()

    return loss


def encoder_iteration(inputs):
    optim_enc.zero_grad()
    x = inputs.transpose(0, 1).to(torch.float32)
    valid_x_score = critic_x(x)
    valid_x_score = torch.squeeze(valid_x_score)
    critic_score_valid_x = torch.mean(valid_x_score)
    z = torch.empty(100, batch_size, latent_space_dim).uniform_(0, 1).to(device)
    x_ = decoder(z)
    fake_x_score = critic_x(x_)
    fake_x_score = torch.squeeze(fake_x_score)
    critic_score_fake_x = torch.mean(fake_x_score)
    enc_z = encoder(x)
    gen_x = decoder(enc_z)
    mse = mse_loss(x.float(), gen_x.float())
    loss_enc = mse - critic_score_fake_x + critic_score_valid_x
    loss_enc.backward()
    optim_enc.step()

    return loss_enc


def decoder_iteration(inputs):
    optim_dec.zero_grad()
    x = inputs.transpose(0, 1).to(torch.float32)
    z = encoder(x)
    fake_z_score = critic_z(z)
    fake_z_score = torch.squeeze(fake_z_score)
    critic_score_valid_z = torch.mean(fake_z_score)

    z_ = torch.empty(100, batch_size, latent_space_dim).uniform_(0, 1).to(device)
    valid_z_score = critic_z(z_)
    valid_z_score = torch.squeeze(valid_z_score)
    critic_score_fake_z = torch.mean(valid_z_score)

    enc_z = encoder(x)
    gen_x = decoder(enc_z)

    mse = mse_loss(x.float(), gen_x.float())
    loss_dec = mse - critic_score_fake_z + critic_score_valid_z
    loss_dec.backward()
    optim_dec.step()

    return loss_dec


def train(n_epochs):
    logging.debug('Starting training')
    cx_epoch_loss = list()
    cz_epoch_loss = list()
    encoder_epoch_loss = list()
    decoder_epoch_loss = list()

    for epoch in range(n_epochs):
        print('======================================================')
        start = time.time()
        logging.debug('Epoch {}'.format(epoch))
        print('Epoch {}'.format(epoch))
        n_critics = 5

        cx_nc_loss = list()
        cz_nc_loss = list()

        for i in range(n_critics):
            cx_loss = list()
            cz_loss = list()
            for batch, (inputs, lables) in enumerate(train_loader):
                inputs = inputs.to(device)
                lables = lables.to(device)
                loss = critic_x_iteration(inputs)
                cx_loss.append(loss)
                loss = critic_z_iteration(inputs)
                cz_loss.append(loss)
            # nc代表n_critics
            cx_nc_loss.append(torch.mean(torch.tensor(cx_loss)))
            cz_nc_loss.append(torch.mean(torch.tensor(cz_loss)))

        encoder_loss = list()
        decoder_loss = list()

        for batch, (inputs, lables) in enumerate(train_loader):
            inputs = inputs.to(device)
            lables = lables.to(device)
            enc_loss = encoder_iteration(inputs)
            dec_loss = decoder_iteration(inputs)
            encoder_loss.append(enc_loss)
            decoder_loss.append(dec_loss)

        cx_epoch_loss.append(torch.mean(torch.tensor(cx_nc_loss)))
        cz_epoch_loss.append(torch.mean(torch.tensor(cz_nc_loss)))
        encoder_epoch_loss.append(torch.mean(torch.tensor(encoder_loss)))
        decoder_epoch_loss.append(torch.mean(torch.tensor(decoder_loss)))
        logging.debug('Encoder decoder training done in epoch {}'.format(epoch))
        logging.debug(
            '\ncritic x loss {:.3f}\ncritic z loss {:.3f} \nencoder loss {:.3f}\ndecoder loss {:.3f}\n'.format(
                cx_epoch_loss[-1], cz_epoch_loss[-1], encoder_epoch_loss[-1], decoder_epoch_loss[-1]))
        print('Encoder decoder training done in epoch {}'.format(epoch))
        print('critic x loss: {:.3f}\ncritic z loss: {:.3f} \nencoder loss: {:.3f}\ndecoder loss: {:.3f}\n'.format(
            cx_epoch_loss[-1], cz_epoch_loss[-1], encoder_epoch_loss[-1], decoder_epoch_loss[-1]))
        end = time.time()
        print('time used: %.2f seconds' % (end - start))
        print('======================================================')
        if (epoch + 1) % 10 == 0:
            torch.save(encoder.state_dict(), encoder.encoder_path)
            torch.save(decoder.state_dict(), decoder.decoder_path)
            torch.save(critic_x.state_dict(), critic_x.critic_x_path)
            torch.save(critic_z.state_dict(), critic_z.critic_z_path)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fPath_Normal_KDD = 'kddcup_normal.csv'
    fPath_Attack_KDD = 'kddcup_test.csv'

    dataset_train, label_train, dataset_test, label_test = process_data_csv(fPath_Normal_KDD, fPath_Attack_KDD)

    train_dataset = SignalDataset(dataset_train, label_train, 100, 50)
    test_dataset = SignalDataset(dataset_test, label_test, 100, 50)
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    lr = 1e-5

    features_shape = 37
    latent_space_dim = 20
    encoder_path = 'models/encoder.pt'
    decoder_path = 'models/decoder.pt'
    critic_x_path = 'models/critic_x.pt'
    critic_z_path = 'models/critic_z.pt'

    encoder = model.Encoder(encoder_path, features_shape)
    decoder = model.Decoder(decoder_path)
    critic_x = model.CriticX(critic_x_path, features_shape)
    critic_z = model.CriticZ(critic_z_path)

    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    critic_x.load_state_dict(torch.load(critic_x_path))
    critic_z.load_state_dict(torch.load(critic_z_path))

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    critic_x = critic_x.to(device)
    critic_z = critic_z.to(device)

    mse_loss = torch.nn.MSELoss()

    optim_enc = optim.Adam(encoder.parameters(), lr=lr, betas=(0, 0.9))
    optim_dec = optim.Adam(decoder.parameters(), lr=lr, betas=(0, 0.9))
    optim_cx = optim.Adam(critic_x.parameters(), lr=lr, betas=(0, 0.9))
    optim_cz = optim.Adam(critic_z.parameters(), lr=lr, betas=(0, 0.9))

    # train(n_epochs=200)

    anomaly_detection.test(test_loader, encoder, decoder, critic_x)
