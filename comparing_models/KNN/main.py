#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import utils
from utils import *
from time import time
import numpy as np
import torch


def KNN_D(k, dataset_train, dataset_test, train, win_size):
    if train:
        print("****************now training!****************")
        dataset_test = dataset_train
    else:
        print("****************now testing!****************")
    train_len = dataset_train.shape[0]
    test_len = dataset_test.shape[0]
    print("train_len:", train_len)
    print("test_len:", test_len)
    KNN_distances = torch.zeros(k, 1).to(device)
    KNN_distances_s = torch.zeros(k, 1).to(device)
    KNN_distances_l = torch.zeros(k, 1).to(device)
    for test_batch in range(test_len // win_size + 1):
        start_time = time()
        test_inputs = dataset_test[slice(test_batch * win_size, (test_batch + 1) * win_size)]
        train_inputs = dataset_train
        train_inputs = torch.from_numpy(train_inputs).to(device)
        test_inputs = torch.from_numpy(test_inputs).to(device)
        ED = EuclideanDistances(train_inputs, test_inputs)
        if train:
            X1, _ = ED.sort(0, False)
            KNN_distance_s = X1[1:(k + 1), :]
            X2, _ = ED.sort(0, True)
            KNN_distance_l = X2[1:(k + 1), :]
            KNN_distances_s = torch.cat([KNN_distances_s, KNN_distance_s], dim=1)
            KNN_distances_l = torch.cat([KNN_distances_l, KNN_distance_l], dim=1)
        else:
            X, _ = ED.sort(0, False)
            KNN_distance = X[:k, :]
            KNN_distances = torch.cat([KNN_distances, KNN_distance], dim=1)
        print("{}/{} window has been finished in {:.2f} seconds".format(test_batch, test_len // win_size, (time() - start_time)))
    if train:
        # 计算阈值
        small = torch.mean(KNN_distances_s[:5, 1:]).item()
        large = torch.mean(KNN_distances_l[:5, 1:]).item()
        threshold = (small * large) ** 0.39
        print("threshold =", threshold)
        return threshold
    else:
        test_means = torch.mean(KNN_distances[:, 1:], dim=0)
        return test_means.cpu().numpy()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fPath_Normal_SWaT = 'SWaT_train.csv'
    fPath_Attack_SWaT = 'SWaT_test.csv'
    fPath_Normal_KDD = 'kddcup_normal.csv'
    fPath_Attack_KDD = 'kddcup_test.csv'
    fPath_Normal_CICIDS = 'cicids_Monday_train.csv'
    fPath_Attack_CICIDS = 'cicids_FridayAfternoon1_test.csv'

    dataset_train, label_train, dataset_test, label_test = process_data_csv(fPath_Normal_SWaT, fPath_Attack_SWaT)
    # dataset_train, label_train, dataset_test, label_test = process_data_csv(fPath_Normal_KDD, fPath_Attack_KDD)
    # dataset_train, label_train, dataset_test, label_test = process_data_csv(fPath_Normal_CICIDS, fPath_Attack_CICIDS)

    threshold = KNN_D(5, dataset_train, dataset_train, True, 50)

    final_distances = KNN_D(5, dataset_train, dataset_test, False, 50)

    test_result = detection(final_distances, threshold=threshold)
    find_scores(label_test, test_result)

