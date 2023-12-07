import os
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from evaluation.event_completion import load_embeds_and_labels, construct_embs_labels_list


def fit_svm_model(train_embs, train_labels, val_embs, val_labels, cal_f1_score=False):
    """Fit a SVM classifier."""
    svm_model = SVC(decision_function_shape='ovo', verbose=False)
    svm_model.fit(train_embs, train_labels)
    train_acc = svm_model.score(train_embs, train_labels)
    val_acc = svm_model.score(val_embs, val_labels)
    if cal_f1_score:
        val_preds = svm_model.predict(val_embs)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        return val_f1
    else:
        return svm_model, train_acc, val_acc


def classification(save_path, train_video_ego_id, val_video_ego_id):
    train_embs, train_labels, val_embs, val_labels = load_embeds_and_labels(save_path)
    regular_f1 = fit_svm_model(train_embs, train_labels, val_embs, val_labels, cal_f1_score=True)

    train_ego_idx = np.array(train_video_ego_id) == 1
    train_exo_idx = np.array(train_video_ego_id) == 0
    val_ego_idx = np.array(val_video_ego_id) == 1
    val_exo_idx = np.array(val_video_ego_id) == 0
    print(f'train: ego frames {np.sum(train_ego_idx)}, exo frames {np.sum(train_exo_idx)} | '
          f'val: ego frames {np.sum(val_ego_idx)}, exo frames {np.sum(val_exo_idx)}')
    ego2exo_val_f1 = fit_svm_model(train_embs[train_ego_idx], train_labels[train_ego_idx],
                                val_embs[val_exo_idx], val_labels[val_exo_idx], cal_f1_score=True)
    exo2ego_val_f1 = fit_svm_model(train_embs[train_exo_idx], train_labels[train_exo_idx],
                                val_embs[val_ego_idx], val_labels[val_ego_idx], cal_f1_score=True)

    return regular_f1, ego2exo_val_f1, exo2ego_val_f1


def select_frame_indices(video_len_list, k, random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)  # fix random seeds so that all baselines evaluate on the same subset
    video_num = len(video_len_list)
    selected_videos = random.sample(range(video_num), k)
    total_frames = sum(video_len_list)
    mask = np.zeros(total_frames, dtype=int)

    for idx in selected_videos:
        start_idx = sum(video_len_list[:idx])
        end_idx = start_idx + video_len_list[idx]
        mask[start_idx:end_idx] = 1

    return mask


def classification_fewshot(save_path, train_video_len_list, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    train_embs, train_labels, val_embs, val_labels = load_embeds_and_labels(save_path)
    k_list = [1, 2, 3, 5, 10, int(0.5 * len(train_video_len_list)), len(train_video_len_list)]
    result = np.zeros((len(k_list), 2), dtype=float)
    for k_idx, k in enumerate(k_list):
        if k_idx == len(k_list) - 2:
            num_episodes = 10
        elif k_idx == len(k_list) - 1:
            num_episodes = 1
        else:
            num_episodes = 50

        if k_idx == len(k_list) - 1:
            num_episodes = 1
        accs = np.zeros((num_episodes, 3), dtype=float)
        for i in range(num_episodes):
            mask = select_frame_indices(train_video_len_list, k, random_seed + i)
            idx1 = np.where(mask == 1)[0]
            svm_model, train_acc, val_acc, val_f1 = fit_svm_model(train_embs[idx1, :], train_labels[idx1],
                                                                  val_embs, val_labels, cal_f1_score=True)
            if k_idx != len(k_list) - 1:
                idx2 = np.where(mask == 0)[0]
                train_acc2 = svm_model.score(train_embs[idx2, :], train_labels[idx2])
            else:
                train_acc2 = 0.0
            accs[i][0] = train_acc2
            accs[i][1] = val_acc
            accs[i][2] = val_f1
            # print(f'run {i}, train acc {train_acc:.4f} | train acc 2 {train_acc2:.4f} |'
            #       f'val acc {val_acc:.4f} | val f1 {val_f1:.4f}')
        accs_mean = np.mean(accs * 100, axis=0)
        accs_std = np.std(accs * 100, axis=0)
        print(f'{k} labeled videos, train (propagate) = {accs_mean[0]:.2f} +- {accs_std[0]:.2f}, '
              f'val = {accs_mean[1]:.2f} +- {accs_std[1]:.2f}, '
              f'val f1 = {accs_mean[2]:.2f} +- {accs_std[2]:.2f}')
        result[k_idx, 0] = round(accs_mean[2], 4)
        result[k_idx, 1] = round(accs_std[2], 4)

    print('[' + ', '.join(['{:.2f}'.format(num) for num in result[:, 0]]) + ']')
    print('[' + ', '.join(['{:.2f}'.format(num) for num in result[:, 1]]) + ']')
    print(k_list)