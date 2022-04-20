import numpy as np
from scipy import stats
import time
import torch
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
l1_loss = torch.nn.L1Loss()
mse_loss = torch.nn.MSELoss()
bce_loss = torch.nn.BCELoss()
lambda_ = 0.5
batch_size = 64


def test(test_loader, generator, discriminator):
    reconstruction_error = list()
    discrimination_score = list()
    y_true_label = list()
    for batch, (inputs, labels) in enumerate(test_loader):
        x = inputs.to(device)
        labels = labels.to(device)
        # z = find_z(x, generator, latent_space_dim=40, max_iter=100)
        z = np.random.uniform(low=-1.0, high=1.0, size=(batch_size, 40))
        z = torch.from_numpy(z).to(torch.float32).to(device)
        reconstructed_x = generator(z)
        reconstruction_error.extend([i.item() for i in [torch.nn.MSELoss()(m, n) for (m, n) in zip(x, reconstructed_x)]])
        discrimination_x = discriminator(x).detach().squeeze().cpu().numpy()
        discrimination_score.extend(discrimination_x)
        y_true_label.extend(labels.detach().cpu().numpy())
    reconstruction_error = np.array(reconstruction_error)
    discrimination_score = np.array(discrimination_score)
    anomaly_score = lambda_ * reconstruction_error + (1 - lambda_) * discrimination_score
    y_predict = detect_anomaly(anomaly_score, 1, 11)
    find_scores(np.array(y_true_label), y_predict)


def detect_anomaly(anomaly_score, epsilon1, epsilon2):
    is_anomaly_threshold_method = np.zeros(len(anomaly_score))
    window_elts = anomaly_score
    for k, elt in enumerate(window_elts):
        if epsilon1 < elt < epsilon2:
            is_anomaly_threshold_method[k] = 0
        else:
            is_anomaly_threshold_method[k] = 1
    is_anomaly = is_anomaly_threshold_method
    return is_anomaly

