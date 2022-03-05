import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normalization_(data_df):
    std_series = data_df.std()
    mean_series = data_df.mean()
    data_df = (data_df - mean_series) / (std_series + 1e-15)
    return data_df


# return train_data_numpy, train_label_numpy, test_data_numpy, test_label_numpy
def process_datacsv_to_normal_distribution(file_path_train, file_path_test):
    train_csv = pd.read_csv(file_path_train)
    test_csv = pd.read_csv(file_path_test)
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


# 计算梯度惩罚wp
def compute_gradient_penalty(critic_x, real_samples, fake_samples):
    """Calculates the gradient penalty loss for GP"""
    # Random weight term for interpolation between real and fake samples 随机插值权重
    alpha = torch.FloatTensor(real_samples.shape).uniform_(0, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(device)
    d_interpolates = critic_x(interpolates)
    weight = torch.Tensor(real_samples.shape[0], real_samples.shape[1]).fill_(1.0).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=weight,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    l2_norm = torch.sqrt(torch.sum(torch.square(gradients)))
    # gradient_penalty = (l2_norm - 1) ** 2
    gradient_penalty = (max(0, (l2_norm - 1)) ** 2)
    return gradient_penalty


# data_tensor是data矩阵
def calculate_Cosine_matirx_intensor(data_tensor):
    numerator = data_tensor.T @ data_tensor
    temp = torch.norm(data_tensor, p=2, dim=0)
    denominator = temp.unsqueeze(1) @ temp.unsqueeze(0)
    return numerator / (denominator + 1e-10)


# 保存各个Cos矩阵
def save_cosine_data(data_numpy, label_numpy, file_path, step=50, win_size=100):
    data_list = list()
    for i in tqdm(range(0, len(data_numpy) - win_size, step)):
        label = label_numpy[i: i + win_size].max()
        data_temp = data_numpy[i: i + win_size, :]
        data_temp = calculate_Cosine_matirx_intensor(torch.from_numpy(data_temp)).numpy()
        data_list.append((data_temp, label))
    with open(file_path, "wb") as f:
        pickle.dump(data_list, f)


if __name__ == '__main__':
    fPath_Attack_v0 = 'cicids_FridayAfternoon1_test.csv'
    fPath_Normal_v0 = 'cicids_Monday_train.csv'
    dataset_train, label_train, dataset_test, label_test = process_datacsv_to_normal_distribution(fPath_Normal_v0, fPath_Attack_v0)
    save_cosine_data(data_numpy=dataset_train, label_numpy=label_train, file_path='cosine_train')
    save_cosine_data(data_numpy=dataset_test, label_numpy=label_test, file_path='cosine_test')
