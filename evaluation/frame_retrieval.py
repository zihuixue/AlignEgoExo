import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def retrieval_ap_at_k(video_len_list, video_paths, embeddings, labels, k_list, cross_view):
    dim = 2 if cross_view else 1
    ap = np.zeros((len(k_list), dim))
    num_queries = np.zeros((len(k_list), dim))

    nbrs = NearestNeighbors(n_neighbors=embeddings.shape[0], algorithm='auto').fit(embeddings)
    # nbrs = NearestNeighbors(n_neighbors=k_list[-1] * 50, algorithm='auto').fit(embeddings)

    frameid2videoid = {}
    cur_idx = 0
    for i, video_file in enumerate(video_paths):
        video_len = video_len_list[i]
        is_ego = True if 'ego' in video_file else False
        for frameid in range(cur_idx, cur_idx + video_len):
            frameid2videoid[frameid] = [i, is_ego, frameid - cur_idx]
        cur_idx = cur_idx + video_len

    for i in tqdm(range(embeddings.shape[0])):
        # Get the query embedding and label
        query_embedding = embeddings[i]
        query_label = labels[i]

        # Find the K+1 nearest neighbors (the first neighbor is the query itself)
        distances, indices = nbrs.kneighbors([query_embedding])
        indices = indices.flatten()

        if cross_view:
            indices = [j for j in indices if
                       frameid2videoid[j][0] != frameid2videoid[i][0]
                       and frameid2videoid[j][1] != frameid2videoid[i][1]]
        else:
            indices = [j for j in indices if frameid2videoid[j][0] != frameid2videoid[i][0]]

        for k_idx, k in enumerate(k_list):
            indices_s = indices[:k].copy()
            assert len(indices_s) == k

            # Count the number of relevant neighbors (with the same label as the query)
            num_relevant = np.sum(labels[indices_s] == query_label)

            # Calculate precision at each rank up to K
            precisions = np.zeros(k)
            for j in range(k):
                precisions[j] = np.sum(labels[indices_s[:j + 1]] == query_label) / (j + 1)

            # Calculate average precision for this query
            if cross_view:
                ego_idx = int(frameid2videoid[i][1])
            else:
                ego_idx = 0
            if num_relevant > 0:
                ap[k_idx][ego_idx] += np.sum(precisions * (labels[indices_s] == query_label)) / num_relevant
            else:
                ap[k_idx][ego_idx] += 0.0
            num_queries[k_idx][ego_idx] += 1

    if cross_view:
        ego2exo = (ap / num_queries)[:, 1]
        exo2ego = (ap / num_queries)[:, 0]
        return ego2exo.squeeze(), exo2ego.squeeze()

    else:
        return (ap / num_queries).squeeze()


def frame_retrieval(save_path, video_len_list, video_paths):
    val_embs = np.load(f'{save_path}/val_embeds.npy')
    val_labels = np.load(f'{save_path}/val_label.npy')
    regular = retrieval_ap_at_k(video_len_list, video_paths, val_embs, val_labels, [10], cross_view=False)
    ego2exo, exo2ego = retrieval_ap_at_k(video_len_list, video_paths, val_embs, val_labels, [10], cross_view=True)
    return regular, ego2exo, exo2ego
