import numpy as np
from scipy import stats
import time
import torch
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(test_loader, encoder, decoder, critic_x):
    start = time.time()
    reconstruction_error = list()
    critic_score = list()
    y_true_lable = list()
    for batch, (inputs, lables) in enumerate(test_loader):
        start_time_batch = time.time()
        inputs = inputs.to(device)
        lables = lables.to(device)
        inputs = inputs.transpose(0, 1).to(torch.float32)
        reconstructed_x = decoder(encoder(inputs))
        reconstructed_x = reconstructed_x.transpose(0, 1).to(torch.float32)

        for i in range(0, 64):
            x_ = reconstructed_x[i].detach().cpu().numpy()
            x = inputs.transpose(0, 1).to(torch.float32)[i].cpu().numpy()
            y_true_lable.append(int(lables[i].detach()))
            reconstruction_error.append(dtw_reconstruction_error(x, x_))
        critic_score.extend(torch.squeeze(critic_x(inputs).mean(0)).detach().cpu().numpy())
        end_time_batch = time.time()
        print('batch: ', batch)
        print('test time for this batch: %.2f seconds' % (end_time_batch - start_time_batch))
    reconstruction_error = stats.zscore(reconstruction_error)
    critic_score = stats.zscore(critic_score)
    anomaly_score = reconstruction_error * critic_score
    y_predict = detect_anomaly(anomaly_score, 0, 6)
    # y_predict = prune_false_positive(y_predict, anomaly_score, change_threshold=0.1)
    find_scores(y_true_lable, y_predict)
    end = time.time()
    print('test time together: %.2f seconds' % (end - start))


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

