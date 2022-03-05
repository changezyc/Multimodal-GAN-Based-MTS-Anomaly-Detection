import torch
import torch.nn as nn


class LSTMAutoEncoder(nn.Module):
    def __init__(self, autoencoder_path, features_shape):
        super(LSTMAutoEncoder, self).__init__()
        self.relu = nn.ReLU()
        self.features_shape = features_shape
        self.lstm1 = nn.LSTM(input_size=self.features_shape, hidden_size=40, num_layers=1, bidirectional=False, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=40, hidden_size=self.features_shape, num_layers=1, bidirectional=False, batch_first=True)
        self.autoencoder_path = autoencoder_path

    def forward(self, x):
        x, (hn, cn) = self.lstm1(x)
        x = self.relu(x)
        x, (hn, cn) = self.lstm2(x)
        return x