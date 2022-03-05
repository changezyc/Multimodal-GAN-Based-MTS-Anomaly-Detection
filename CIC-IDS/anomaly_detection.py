import numpy as np
from scipy import stats
import time
import torch
import utils
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64
lambda_ = 0.99
theta = 1


def test(test_loader_seq, generator_rnn, generator_cnn, discriminator, encoder_rnn, encoder_cnn):
    start = time.time()
    reconstruction_error_time = list()
    reconstruction_error_space = list()
    dis_score = list()
    anomaly_score = list()
    y_true_label = list()
    for batch, (inputs, labels) in enumerate(test_loader_seq):
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.to(torch.float32)
        reconstructed_x_time = generator_rnn(encoder_rnn(inputs))
        reconstructed_x_time = reconstructed_x_time.to(torch.float32)
        x_r = inputs
        x_r_c = torch.empty(batch_size, 77, 77).to(device)
        for i in range(len(inputs)):
            x_r_c[i] = utils.calculate_Cosine_matirx_intensor(x_r[i])
        x_r_c = x_r_c.unsqueeze(1)
        reconstructed_x_space = generator_cnn(encoder_cnn(x_r_c))
        reconstructed_x_space = reconstructed_x_space.to(torch.float32)

        x_cos = torch.empty(batch_size, 77, 77).to(device)
        for i in range(len(inputs)):
            x_cos[i] = utils.calculate_Cosine_matirx_intensor(inputs[i])
        x_cos = x_cos.unsqueeze(1).to(torch.float32)
        y_true_label.extend((labels.detach()).int())
        reconstruction_error_time.extend(
            [i.item() for i in [torch.nn.MSELoss()(m, n) for (m, n) in zip(inputs, reconstructed_x_time)]])

        reconstruction_error_space.extend(
            [i.item() for i in [torch.nn.MSELoss()(m, n) for (m, n) in zip(x_cos, reconstructed_x_space)]])

        dis_each_x = discriminator(x_cos).detach().squeeze()
        dis_score.extend([i.item() for i in dis_each_x])


    anomaly_score_rec_time = np.array(reconstruction_error_time)
    anomaly_score_rec_space = np.array(reconstruction_error_space)
    anomaly_score_rec = theta * anomaly_score_rec_time + (1 - theta) * anomaly_score_rec_space
    anomaly_score_dis = np.array(dis_score)

    y_predict_rec = detect_anomaly(anomaly_score=anomaly_score_rec, epsilon1=0, epsilon2=0.6, method=2)
    # y_predict_rec = prune_single_positive(is_anomaly=y_predict_rec)
    y_predict_dis = detect_anomaly(anomaly_score=anomaly_score_dis, epsilon1=0.8, epsilon2=1.8, method=2)
    # y_predict_dis = prune_single_positive(is_anomaly=y_predict_dis)

    temp_anoscore = (1 - lambda_) * anomaly_score_rec + lambda_ * np.array(dis_score)
    temp_anoscore.tolist()
    anomaly_score.extend(temp_anoscore)

    # np.savetxt('./anomaly_score_rec.txt', anomaly_score_rec, delimiter=',')
    # np.savetxt('./anomaly_score_dis.txt', anomaly_score_dis, delimiter=',')
    # np.savetxt('./y_true_label.txt', (np.array(y_true_label)).astype(int), delimiter=',')
    # np.savetxt('./anomaly_score.txt', anomaly_score, delimiter=',')

    y_predict_together = detect_anomaly(anomaly_score, epsilon1=0.84, epsilon2=2.05, method=2)
    y_predict_together = prune_single_positive(is_anomaly=y_predict_together)
    find_scores(y_true_label, y_predict_together)
    print('========================================================')
    print('========================================================')
    print('重构：')
    find_scores(y_true_label, y_predict_rec)
    print('========================================================')
    print('判别：')
    find_scores(y_true_label, y_predict_dis)
    end = time.time()
    print('test time together: %.2f seconds' % (end - start))


def prune_single_positive(is_anomaly):

    for i in range(1, len(is_anomaly) - 1):
        if is_anomaly[i] == 1 and is_anomaly[i + 1] == 0 and is_anomaly[i - 1] == 0:
            is_anomaly[i] = 0

    return is_anomaly


def detect_anomaly(anomaly_score, epsilon1, epsilon2, method):

    is_anomaly_threshold_method = np.zeros(len(anomaly_score))
    window_elts = anomaly_score
    if method == 1:
        for k, elt in enumerate(window_elts):
            if epsilon1 < elt < epsilon2:
                is_anomaly_threshold_method[k] = 0
            else:
                is_anomaly_threshold_method[k] = 1
        is_anomaly = is_anomaly_threshold_method
    else:
        for k, elt in enumerate(window_elts):
            if epsilon1 < elt < epsilon2:
                is_anomaly_threshold_method[k] = 1
            else:
                is_anomaly_threshold_method[k] = 0
        is_anomaly = is_anomaly_threshold_method

    return is_anomaly


def find_scores(y_true, y_predict):
    tp = tn = fp = fn = 0

    for i in range(0, len(y_true)):
        if y_true[i] == 1 and y_predict[i] == 1:
            tp += 1
        elif y_true[i] == 1 and y_predict[i] == 0:
            fn += 1
        elif y_true[i] == 0 and y_predict[i] == 0:
            tn += 1
        elif y_true[i] == 0 and y_predict[i] == 1:
            fp += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('Precision {:.4f}'.format(precision))
    print('Recall {:.4f}'.format(recall))
    print('F1 Score {:.4f}'.format(2 * precision * recall / (precision + recall)))
