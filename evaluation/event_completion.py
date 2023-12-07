import os
import sklearn
import numpy as np
from sklearn.metrics import mean_squared_error


class VectorRegression(sklearn.base.BaseEstimator):
    """Class to perform regression on multiple outputs."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, x, y):
        _, m = y.shape
        # Fit a separate regressor for each column of y
        self.estimators_ = []
        for i in range(m):
            idx = np.where(y[:, i] != -1)[0]
            x_idx = x[idx]
            y_idx = y[idx, i]
            self.estimators_.append(sklearn.base.clone(self.estimator).fit(x_idx, y_idx))
        return self

    def predict(self, x):
        # Join regressors' predictions
        res = [est.predict(x)[:, np.newaxis] for est in self.estimators_]
        return np.hstack(res)

    def score(self, x, y):
        # Join regressors' scores
        res = []
        for i, est in enumerate(self.estimators_):
            idx = np.where(y[:, i] != -1)[0]
            x_idx = x[idx]
            y_idx = y[idx, i]
            res.append(est.score(x_idx, y_idx))
        # print('score', res)
        return np.mean(res)


def load_embeds_and_labels(save_path):
    train_embs = np.load(f'{save_path}/train_embeds.npy')
    train_labels = np.load(f'{save_path}/train_label.npy')
    val_embs = np.load(f'{save_path}/val_embeds.npy')
    val_labels = np.load(f'{save_path}/val_label.npy')
    return train_embs, train_labels, val_embs, val_labels


def construct_embs_labels_list(embs, labels, video_len_list, modify_embeddings, return_list=False):
    cur_idx = 0
    embs_list, labels_list = [], []
    for i in range(len(video_len_list)):
        video_len = video_len_list[i]
        embs_tmp = embs[cur_idx: cur_idx + video_len, :]
        if modify_embeddings:
            col = np.arange(video_len).reshape(-1, 1) * 1e-3  # augment the embeddings with a temporal dimension
            embs_tmp = np.concatenate((embs_tmp, col), axis=1)
        embs_list.append(embs_tmp)
        labels_list.append(labels[cur_idx: cur_idx + video_len])
        cur_idx = cur_idx + video_len
    if return_list:
        return embs_list, labels_list
    labels_list = get_targets_from_labels(labels_list, num_classes=int(max(labels)) + 1)
    embs = np.concatenate(embs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return embs, labels


def regression_labels_for_class(labels, class_idx):
    # Assumes labels are ordered. Find the last occurrence of particular class.
    if class_idx not in labels:
        return -1 * np.ones(len(labels))
        # assert class_idx - 1 in labels
        # return regression_labels_for_class(labels, class_idx-1)  #-1 * np.ones(len(labels))
    transition_frame = np.argwhere(labels == class_idx)[-1, 0]
    return (np.arange(float(len(labels))) - transition_frame) / len(labels)


def get_regression_labels(class_labels, num_classes):
    regression_labels = []
    for i in range(num_classes):
        regression_labels.append(regression_labels_for_class(class_labels, i))
    return np.stack(regression_labels, axis=1)


def get_targets_from_labels(all_class_labels, num_classes):
    all_regression_labels = []
    for class_labels in all_class_labels:
        all_regression_labels.append(get_regression_labels(class_labels,
                                                           num_classes))
    return all_regression_labels


def compute_progression_value(save_path, train_video_len_list, val_video_len_list, modify_embeddings=False):
    train_embs, train_labels, val_embs, val_labels = load_embeds_and_labels(save_path)
    train_embs, train_labels = construct_embs_labels_list(train_embs, train_labels, train_video_len_list, modify_embeddings)
    val_embs, val_labels = construct_embs_labels_list(val_embs, val_labels, val_video_len_list, modify_embeddings)

    lin_model = VectorRegression(sklearn.linear_model.LinearRegression())
    lin_model.fit(train_embs, train_labels)
    train_score = lin_model.score(train_embs, train_labels)
    val_score = lin_model.score(val_embs, val_labels)
    return train_score, val_score



