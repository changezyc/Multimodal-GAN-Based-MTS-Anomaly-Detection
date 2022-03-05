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
        print("===============================")
        print("分母为0，无法计算!")
        return 0

