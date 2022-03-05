#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

features_shape = 77


class Generator(nn.Module):
    def __init__(self, generator_path, features_shape):
        super(Generator, self).__init__()
        self.features_shape = features_shape
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        self.dense_r_1 = nn.Linear(in_features=40, out_features=640)
        self.dense_r_2 = nn.Linear(in_features=640, out_features=4000)
        self.lstm = nn.LSTM(input_size=40, hidden_size=128, num_layers=1, batch_first=True)
        self.dense_r_3 = nn.Linear(in_features=128, out_features=features_shape)
        self.sigmoid = nn.Sigmoid()
        self.generator_path = generator_path

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.dense_r_1(x)
        x = self.leakyrelu(x)
        x = self.dense_r_2(x)
        x = self.leakyrelu(x)
        x = x.view(-1, 100, 40)
        x, (hn, cn) = self.lstm(x)
        x = self.leakyrelu(x)
        x = self.dense_r_3(x)
        x = self.tanh(x)
        # x = self.sigmoid(x)
        return x


class Discriminator(nn.Module):

    def __init__(self, discriminator_path, features_shape):
        super(Discriminator, self).__init__()
        self.features_shape = features_shape
        # self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=77, hidden_size=40, num_layers=1, batch_first=True)
        self.dense1 = nn.Linear(in_features=4000, out_features=640)
        self.dense2 = nn.Linear(in_features=640, out_features=40)
        self.dense3 = nn.Linear(in_features=40, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.discriminator_path = discriminator_path

    def forward(self, x):
        x = x.to(torch.float32)
        x, (hn, cn) = self.lstm(x)
        x = x.contiguous().view(-1, 4000)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        x = self.sigmoid(x)
        return x