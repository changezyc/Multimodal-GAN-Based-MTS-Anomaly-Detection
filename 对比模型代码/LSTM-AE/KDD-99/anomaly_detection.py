import numpy as np
import torch
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
l1_loss = torch.nn.L1Loss()
mse_loss = torch.nn.MSELoss()


def test(test_loader, autoencoder):
    reconstruction_error = list()
    y_true_label = list()
    for batch, (inputs, labels) in enumerate(test_loader):
        x = inputs.float().to(device)
        labels = labels.to(device)
        reconstructed_x = autoencoder(x)
        reconstruction_error.extend([i.item() for i in [torch.nn.MSELoss()(m, n) for (m, n) in zip(x, reconstructed_x)]])
        y_true_label.extend(labels.detach().cpu().numpy())
    reconstruction_error = np.array(reconstruction_error)
    y_predict = detect_anomaly(reconstruction_error, 0, 0.045)
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

