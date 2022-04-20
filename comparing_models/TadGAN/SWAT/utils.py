import pandas as pd
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normalization_(data_df):
    std_series = data_df.std()
    mean_series = data_df.mean()
    data_df = (data_df - mean_series) / (std_series + 1e-10)
    return data_df


# return train_data_numpy, train_label_numpy, test_data_numpy, test_label_numpy
def process_data_csv(file_path_train, file_path_test):
    train_csv = pd.read_csv(file_path_train)
    test_csv = pd.read_csv(file_path_test)
    if file_path_train == 'kddcup_normal.csv' and file_path_test == 'kddcup_test.csv':
        train_label_csv = train_csv.iloc[:, -1]
        train_data_csv = train_csv.iloc[:, 4:-1]
        test_label_csv = test_csv.iloc[:, -1]
        test_data_csv = test_csv.iloc[:, 4:-1]
    else:
        train_label_csv = train_csv.iloc[:, -1]
        train_data_csv = train_csv.iloc[:, 1:-1]
        test_label_csv = test_csv.iloc[:, -1]
        test_data_csv = test_csv.iloc[:, 1:-1]
    train_data_csv = normalization_(train_data_csv)
    test_data_csv = normalization_(test_data_csv)
    train_data_numpy = train_data_csv.to_numpy()
    test_data_numpy = test_data_csv.to_numpy()
    train_label_numpy = train_label_csv.to_numpy()
    test_label_numpy = test_label_csv.to_numpy()

    return train_data_numpy, train_label_numpy, test_data_numpy, test_label_numpy


def compute_gradient_penalty(Discriminator, real_samples, fake_samples):
    # 随机插值权重
    alpha = torch.rand(real_samples.shape).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(device)
    d_interpolates = Discriminator(interpolates)
    d_interpolates = torch.squeeze(d_interpolates)
    weight = torch.Tensor(real_samples.shape[0], real_samples.shape[1]).fill_(1.0).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=weight,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    l2_norm = torch.sqrt(torch.sum(torch.square(gradients)) / torch.numel(gradients))
    # l2_norm = torch.sqrt(torch.sum(torch.square(gradients)))
    gradient_penalty = ((l2_norm - 1) ** 2)
    return gradient_penalty


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
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print('Precision {:.4f}'.format(precision))
        print('Recall {:.4f}'.format(recall))
        print('F1 Score {:.4f}'.format(2 * precision * recall / (precision + recall)))
    except ZeroDivisionError:
        print("无解！")


def prune_false_positive(is_anomaly, anomaly_score, change_threshold):
    # 模型可能检测到大量的假阳性(FP)。
    # 在这种情况下，建议修剪假阳性(FP)。
    # 使用的方法如第5节D部分“识别异常”所述
    # 序列，子部分-减少误报
    seq_details = []
    delete_sequence = 0
    start_position = 0
    max_seq_element = anomaly_score[0]
    for i in range(1, len(is_anomaly)):
        if i + 1 == len(is_anomaly):
            seq_details.append([start_position, i, max_seq_element, delete_sequence])
        elif is_anomaly[i] == 1 and is_anomaly[i + 1] == 0:
            end_position = i
            seq_details.append([start_position, end_position, max_seq_element, delete_sequence])
        elif is_anomaly[i] == 1 and is_anomaly[i - 1] == 0:
            start_position = i
            max_seq_element = anomaly_score[i]
        if is_anomaly[i] == 1 and is_anomaly[i - 1] == 1 and anomaly_score[i] > max_seq_element:
            max_seq_element = anomaly_score[i]

    max_elements = list()
    for i in range(0, len(seq_details)):
        max_elements.append(seq_details[i][2])

    max_elements.sort(reverse=True)
    max_elements = np.array(max_elements)
    change_percent = abs(max_elements[1:] - max_elements[:-1]) / max_elements[1:]

    delete_seq = np.append(np.array([0]), change_percent < change_threshold)

    for i, max_elt in enumerate(max_elements):
        for j in range(0, len(seq_details)):
            if seq_details[j][2] == max_elt:
                seq_details[j][3] = delete_seq[i]

    for seq in seq_details:
        if seq[3] == 1:
            is_anomaly[seq[0]:seq[1] + 1] = [0] * (seq[1] - seq[0] + 1)

    return is_anomaly


# x和x_是两个numpy数组，dtw算法
def dtw_reconstruction_error(x, x_):

    n, m = x.shape[0], x_.shape[0]
    dtw_matrix = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.sqrt(np.sum(np.square(x[i - 1] - x_[j - 1])))
            # take last min from a square box
            last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[n][m]