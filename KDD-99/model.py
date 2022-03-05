#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

features_shape = 37


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.selu = nn.SELU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(2, 2), stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=0)
        self.dense1 = nn.Linear(in_features=2048, out_features=100)
        self.dense2 = nn.Linear(in_features=100, out_features=1, bias=False)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.conv1(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.leakyrelu(x)
        x = self.conv3(x)
        x = self.leakyrelu(x)
        x = x.view(-1, 2048)
        x = self.dense1(x)
        x = self.leakyrelu(x)
        x = self.dense2(x)
        return x


# g_cnn
class Generator_C(nn.Module):
    def __init__(self):
        super(Generator_C, self).__init__()
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.selu = nn.SELU()
        self.tanh = nn.Tanh()
        self.dense_c_1 = nn.Linear(in_features=40, out_features=100)
        self.dense_c_2 = nn.Linear(in_features=100, out_features=2048)
        self.conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=2, padding=0)
        self.conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(2, 2), stride=2, padding=0)
        self.conv3 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(3, 3), stride=2, padding=0, bias=False)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.dense_c_1(x)
        x = self.leakyrelu(x)
        # x = self.selu(x)
        x = self.dense_c_2(x)
        x = self.leakyrelu(x)
        x = x.view(-1, 128, 4, 4)
        x = self.conv1(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.leakyrelu(x)
        x = self.conv3(x)
        x = self.tanh(x)
        return x


# g_rnn
class Generator_R(nn.Module):
    def __init__(self):
        super(Generator_R, self).__init__()
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.selu = nn.SELU()
        self.tanh = nn.Tanh()
        self.dense_r_1 = nn.Linear(in_features=40, out_features=640)
        self.dense_r_2 = nn.Linear(in_features=640, out_features=4000)
        self.lstm = nn.LSTM(input_size=40, hidden_size=128, num_layers=1, batch_first=True)
        self.dense_r_3 = nn.Linear(in_features=128, out_features=37)

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
        return x


class Encoder_R(nn.Module):
    def __init__(self):
        super(Encoder_R, self).__init__()
        # # self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=37, hidden_size=40, num_layers=1, batch_first=True)
        self.dense1 = nn.Linear(in_features=4000, out_features=640)
        self.dense2 = nn.Linear(in_features=640, out_features=40)

    def forward(self, x):
        x = x.to(torch.float32)
        x, (hn, cn) = self.lstm(x)
        x = x.contiguous().view(-1, 4000)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x


# encoder_cnn
class Encoder_C(nn.Module):

    def __init__(self):
        super(Encoder_C, self).__init__()
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(2, 2), stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=0)
        self.dense1 = nn.Linear(in_features=2048, out_features=100)
        self.dense2 = nn.Linear(in_features=100, out_features=40, bias=False)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.conv1(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.leakyrelu(x)
        x = self.conv3(x)
        x = self.leakyrelu(x)
        x = x.view(-1, 2048)
        x = self.dense1(x)
        x = self.leakyrelu(x)
        x = self.dense2(x)
        return x


if __name__ == '__main__':
    pass