import numpy as np
import pandas as pd


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


def get_recscore(x, eig_pair):
    score = 0
    for i in range(len(eig_pair)):
        if eig_pair[i][0] == 0:
            continue
        temp = ((x.dot(eig_pair[i][1][:, np.newaxis])) ** 2).squeeze() / eig_pair[i][0]
        score = score + temp
    return score


def get_eigs(dataset):

    cov_mat = np.cov(dataset.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_vals = eig_vals.real
    eig_vecs = eig_vecs.real
    eig_pair = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pair.sort(key=lambda x: x[0], reverse=True)
    return eig_pair


def detection(score, threshold):
    is_anomaly = np.zeros(len(score))
    for i, distance in enumerate(score):
        if score[i] <= threshold:
            is_anomaly[i] = 0
        else:
            is_anomaly[i] = 1
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
    print("==============================")


def get_score(data_name):
    if data_name == "swat" or data_name == "SWAT" or data_name == "SWaT_test.csv":
        print("testing swat!")
        dataset_test = dataset_test_swat
        score_np = get_score_gen(dataset_test=dataset_test)
    elif data_name == "kdd" or data_name == "KDD" or data_name == "kddcup_test.csv":
        print("testing kdd!")
        dataset_test = dataset_test_kdd
        score_np = get_score_gen(dataset_test=dataset_test)
    elif data_name == "cicids" or data_name == "CICIDS" or data_name == "cicids_FridayAfternoon1_test.csv":
        print("testing cicids!")
        dataset_test = dataset_test_cicids
        score_np = get_score_gen(dataset_test=dataset_test)
    else:
        print("error!")
        return
    return score_np


def get_score_gen(dataset_test):
    eig_pair = get_eigs(dataset=dataset_test)
    score_list = []
    for i in range(len(dataset_test)):
        x = dataset_test[i:i + 1, :]
        score = get_recscore(x=x, eig_pair=eig_pair)
        print("========{}=======".format(i + 1))
        print("score = {:.5f}".format(score))
        score_list.append(score)
    score_np = np.array(score_list)
    return score_np

if __name__ == '__main__':

    fPath_Normal_SWaT = 'SWaT_train.csv'
    fPath_Attack_SWaT = 'SWaT_test.csv'
    fPath_Normal_KDD = 'kddcup_normal.csv'
    fPath_Attack_KDD = 'kddcup_test.csv'
    fPath_Normal_CICIDS = 'cicids_Monday_train.csv'
    fPath_Attack_CICIDS = 'cicids_FridayAfternoon1_test.csv'

    dataset_train_swat, label_train_swat, dataset_test_swat, label_test_swat = process_data_csv(fPath_Normal_SWaT, fPath_Attack_SWaT)
    dataset_train_kdd, label_train_kdd, dataset_test_kdd, label_test_kdd = process_data_csv(fPath_Normal_KDD, fPath_Attack_KDD)
    dataset_train_cicids, label_train_cicids, dataset_test_cicids, label_test_cicids = process_data_csv(fPath_Normal_CICIDS, fPath_Attack_CICIDS)

    threshold_swat = 29
    threshold_kdd = 1
    threshold_cicids = 10

    score_np_swat = get_score("swat")
    score_np_kdd = get_score("kdd")
    score_np_cicids = get_score("cicids")

    predict_swat = detection(score=score_np_swat, threshold=threshold_swat)
    predict_kdd = detection(score=score_np_kdd, threshold=threshold_kdd)
    predict_cicids = detection(score=score_np_cicids, threshold=threshold_cicids)

    print("swat:")
    find_scores(label_test_swat, predict_swat)
    print("kdd:")
    find_scores(label_test_kdd, predict_kdd)
    print("cicids:")
    find_scores(label_test_cicids, predict_cicids)