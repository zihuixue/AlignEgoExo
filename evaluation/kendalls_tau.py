import os
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau


def get_kendalls_tau(embs_list, stride=5):
    """Get nearest neighbours in embedding space and calculate Kendall's Tau."""
    num_seqs = len(embs_list)
    taus = np.zeros((num_seqs * (num_seqs - 1)))
    idx = 0
    for i in range(num_seqs):
        query_feats = embs_list[i][::stride]
        for j in range(num_seqs):
            if i == j:
                continue
            candidate_feats = embs_list[j][::stride]
            dists = cdist(query_feats, candidate_feats, 'sqeuclidean')
            nns = np.argmin(dists, axis=1)
            taus[idx] = kendalltau(np.arange(len(nns)), nns).correlation
            # print(i, j, taus[idx])
            idx += 1
    # Remove NaNs.
    taus = taus[~np.isnan(taus)]
    tau = np.mean(taus)
    return tau


def get_kendalls_tau_twolists(embs_list1, embs_list2, same_view=False, stride=5):
    num_seqs1 = len(embs_list1)
    num_seqs2 = len(embs_list2)
    taus = np.zeros((num_seqs1 * (num_seqs2 - 1))) if same_view else np.zeros((num_seqs1 * num_seqs2))

    idx = 0
    for i in range(len(embs_list1)):
        query_feats = embs_list1[i][::stride]
        for j in range(len(embs_list2)):
            if same_view and i == j:
                continue
            candidate_feats = embs_list2[j][::stride]
            dists = cdist(query_feats, candidate_feats, 'sqeuclidean')
            nns = np.argmin(dists, axis=1)
            taus[idx] = kendalltau(np.arange(len(nns)), nns).correlation
            idx += 1
    taus = taus[~np.isnan(taus)]
    tau = np.mean(taus)
    return tau


def kendalls_tau(save_path, video_len_list, video_paths, mode, detailed_view=False):
    embs = np.load(f'{save_path}/{mode}_embeds.npy')

    cur_idx = 0
    ego_embs_list, exo_embs_list = [], []
    embs_list = []
    for i in range(len(video_len_list)):
        video_len = video_len_list[i]
        tmp = embs[cur_idx: cur_idx + video_len, :]
        if 'ego' in video_paths[i]:
            ego_embs_list.append(tmp)
        else:
            exo_embs_list.append(tmp)
        embs_list.append(tmp)
        cur_idx = cur_idx + video_len

    if detailed_view:
        print(len(ego_embs_list), len(exo_embs_list), len(embs_list))
        print(f'Ego-Ego Kendall Tau {get_kendalls_tau_twolists(ego_embs_list, ego_embs_list, True):.4f}')
        print(f'Exo-Exo Kendall Tau {get_kendalls_tau_twolists(exo_embs_list, exo_embs_list, True):.4f}')
        print(f'Ego-Exo Kendall Tau {get_kendalls_tau_twolists(ego_embs_list, exo_embs_list, False):.4f}')
        print(f'Exo-Ego Kendall Tau {get_kendalls_tau_twolists(exo_embs_list, ego_embs_list, False):.4f}')

    tau = get_kendalls_tau(embs_list)
    return tau

