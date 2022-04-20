#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd
import torch
import numpy as np


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


def EuclideanDistances(A, B):

    BT = torch.transpose(B, 0, 1)
    vecProd = torch.mm(A, BT)
    SqA = A ** 2
    sumSqA = torch.sum(SqA, 1)
    sumSqAEx = sumSqA.unsqueeze(1).repeat(1, vecProd.shape[1])
    SqB = B ** 2
    sumSqB = torch.sum(SqB, 1)
    sumSqBEx = sumSqB.repeat(vecProd.shape[0], 1)
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = torch.sqrt(SqED)

    return ED


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

    print('Accuracy {:.4f}'.format((tp + tn) / (len(y_true))))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('Precision {:.4f}'.format(precision))
    print('Recall {:.4f}'.format(recall))
    print('F1 Score {:.4f}'.format(2 * precision * recall / (precision + recall)))


def detection(final_distances, threshold):
    is_anomaly = np.zeros(len(final_distances))
    for i, distance in enumerate(final_distances):
        if final_distances[i] <= threshold:
            is_anomaly[i] = 0
        else:
            is_anomaly[i] = 1
    return is_anomaly


if __name__ == '__main__':
    pass