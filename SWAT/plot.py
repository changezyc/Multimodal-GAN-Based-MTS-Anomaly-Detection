import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def process_data_csv(file_path_train, file_path_test):
    train_csv = pd.read_csv(file_path_train)
    test_csv = pd.read_csv(file_path_test)
    train_label_csv = train_csv.iloc[:, -1]
    train_data_csv = train_csv.iloc[:, 1:-1]
    test_label_csv = test_csv.iloc[:, -1]
    test_data_csv = test_csv.iloc[:, 1:-1]
    train_data_numpy = train_data_csv.to_numpy()
    test_data_numpy = test_data_csv.to_numpy()
    train_label_numpy = train_label_csv.to_numpy()
    test_label_numpy = test_label_csv.to_numpy()

    return train_data_numpy, train_label_numpy, test_data_numpy, test_label_numpy


fPath_Attack_v0 = 'SWaT_test.csv'
fPath_Normal_v0 = 'SWaT_train.csv'
dataset_train, label_train, dataset_test, label_test = process_data_csv(fPath_Normal_v0, fPath_Attack_v0)
plt.plot(dataset_train)
plt.xlim(0, 2000)
plt.ylim(0, 600)
plt.show()
