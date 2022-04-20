#!/usr/bin/env python
# coding: utf-8

import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, encoder_path, features_shape):
        super(Encoder, self).__init__()
        self.features_shape = features_shape
        self.lstm = nn.LSTM(input_size=self.features_shape, hidden_size=20, num_layers=1, bidirectional=True)
        self.dense = nn.Linear(in_features=40, out_features=20)
        self.encoder_path = encoder_path

    def forward(self, x):
        # x = x.float()
        x, (hn, cn) = self.lstm(x)
        x = self.dense(x)
        return x


class Decoder(nn.Module):
    def __init__(self, decoder_path):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=20, hidden_size=32, num_layers=1, bidirectional=True)
        self.dense = nn.Linear(in_features=64, out_features=51)
        self.decoder_path = decoder_path

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = self.dense(x)
        return x


class CriticX(nn.Module):
    def __init__(self, critic_x_path, features_shape):
        super(CriticX, self).__init__()
        self.features_shape = features_shape
        self.dense1 = nn.Linear(in_features=self.features_shape, out_features=20)
        self.dense2 = nn.Linear(in_features=20, out_features=1)
        self.critic_x_path = critic_x_path

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class CriticZ(nn.Module):
    def __init__(self, critic_z_path):
        super(CriticZ, self).__init__()
        self.dense1 = nn.Linear(in_features=20, out_features=1)
        self.critic_z_path = critic_z_path

    def forward(self, x):
        x = self.dense1(x)
        return x

