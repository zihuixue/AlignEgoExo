import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def classification_loss(logits, labels, label_smoothing):
    """Classification loss """
    # stop gradients from labels
    labels = labels.detach()
    num_classes = logits.size(1)
    smooth_labels = (1 - label_smoothing) * labels + label_smoothing / num_classes
    # cls_loss = F.kl_div(F.log_softmax(logits, dim=1), smooth_labels, reduction='batchmean')
    log_prob = F.log_softmax(logits, dim=1)
    cls_loss = -(smooth_labels * log_prob).sum(dim=1).mean()
    return cls_loss


def assign2Tensor(tensor, i, j, new_val):
    tensor = tensor.clone()
    tensor[i, j] = new_val
    return tensor


def minGamma(inputs, gamma=1):
    if gamma == 0:
        minG = torch.min(inputs)
    else:
        zi = (-inputs / gamma)
        max_zi = torch.max(zi)
        log_sum_G = max_zi + torch.logsumexp(zi - max_zi, dim=-1)
        minG = -gamma * log_sum_G
    return minG


def smoothDTW(embs1, embs2, distance_type, softning, gamma_s, gamma_f, d2tw_norm=True):
    """ function to obtain a soft (differentiable version of DTW) """
    # first get a pairwise distance matrix
    if distance_type == 'cosine':
        dist = torch.matmul(embs1, embs2.transpose(1, 0))
    else:
        raise ValueError('distance_type %s not supported for now' % distance_type)

    # normalize distance column-wise
    if d2tw_norm:
        dist = -torch.nn.functional.log_softmax(dist / gamma_f, dim=0)

    nrows, ncols = dist.shape

    # calculate soft-DTW table
    sdtw = torch.zeros((nrows + 1, ncols + 1), dtype=torch.float32)

    # obtain dtw table using min_gamma or prob relaxation
    for i in range(0, nrows + 1):
        for j in range(0, ncols + 1):
            if (i == 0) and (j == 0):
                new_val = 0.0
                sdtw = assign2Tensor(sdtw, i, j, new_val)
            elif (i == 0) and (j != 0):
                new_val = torch.finfo(torch.float32).max
                sdtw = assign2Tensor(sdtw, i, j, new_val)
            elif (i != 0) and (j == 0):
                new_val = torch.finfo(torch.float32).max
                sdtw = assign2Tensor(sdtw, i, j, new_val)
            else:
                neighbors = torch.stack([sdtw[i, j - 1], sdtw[i - 1, j - 1], sdtw[i - 1, j]])
                if softning == 'dtw_minGamma':
                    new_val = dist[i - 1, j - 1] + minGamma(neighbors, gamma_s)
                    sdtw = assign2Tensor(sdtw, i, j, new_val)
                elif softning == 'dtw_prob':
                    probs = torch.nn.functional.softmax((-neighbors) / gamma_s, dim=-1)
                    new_val = dist[i - 1, j - 1] + (probs[0] * sdtw[i, j - 1]) + (probs[1] * sdtw[i - 1, j - 1]) + (
                            probs[2] * sdtw[i - 1, j])
                    sdtw = assign2Tensor(sdtw, i, j, new_val)
                elif softning == 'non-diff':
                    new_val = dist[i - 1, j - 1] + torch.min(
                        torch.stack([sdtw[i, j - 1], sdtw[i - 1, j - 1], sdtw[i - 1, j]]))
                    sdtw = assign2Tensor(sdtw, i, j, new_val)
                else:
                    raise ValueError('only softning based on dtw_minGamma or dtw_prob supported for now.')
    return sdtw, dist


def compute_dtw_alignment_loss(embs,
                               distance_type,
                               softning,
                               gamma_s,
                               gamma_f
                               ):
    # embs - [batch_size, 2, num_steps, dim]

    batch_size = embs.shape[0]
    logits_list = []
    for i in range(batch_size):
        # if i >= batch_size / 2:
        #     continue
        embs1 = embs[i][0]
        embs2 = embs[i][1]
        logits, _ = smoothDTW(embs1, embs2, distance_type, softning, gamma_s, gamma_f)
        logits_list.append(logits[-1, -1])
    logits = torch.stack(logits_list, dim=0)
    # calculate the loss
    loss = torch.mean(logits)
    return loss

def compute_dtw_alignment_consistency_loss(embs,
                                           distance_type,
                                           softning,
                                           gamma_s,
                                           gamma_f,
                                           label_smoothing,
                                           revcons
                                           ):
    """Compute d2tw loss with Global Cycle Consistency for all steps in each sequence.
    Args:
      embs: Tensor, sequential embeddings of the shape [N, T, D] where N is the
        batch size, T is the number of timesteps in the sequence, D is the size
        of the embeddings.
      loss_type: define the loss type used in our dtw alignment
      distance_type: String, Currently supported distance metrics: 'cosine'
      softning: relaxation used for dtw. currently supported: 'dtw_minGamma' and 'dtw_prob'
    Returns:
      loss: Tensor, Scalar loss tensor that imposes the chosen variant of the
          dtw loss.
    """

    logits_list = []
    logits_ij_list = []
    logits_ji_list = []
    labels_list = []
    batch_size = embs.shape[0]
    for i in range(batch_size):
        embs1 = embs[i][0]
        embs2 = embs[i][1]
        # if i >= batch_size / 2:
        #     continue
        logits_ij, _ = smoothDTW(embs1, embs2, distance_type, softning, gamma_s, gamma_f)
        logits_ij_list.append(logits_ij[-1, -1])
        logits_ij = F.softmax(-logits_ij[1:, 1:], dim=0)

        logits_ji, _ = smoothDTW(embs2, embs1, distance_type, softning, gamma_s, gamma_f)
        logits_ji_list.append(logits_ji[-1, -1])
        logits_ji = F.softmax(-logits_ji[1:, 1:], dim=0)

        if revcons:
            logits = torch.matmul(logits_ij, logits_ji)
            logits = torch.transpose(logits, 0, 1)
            logits_list.append(logits)
            labels = torch.eye(logits.shape[0])
            labels_list.append(labels)

    if revcons:
        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

    logits_ij_list = torch.stack(logits_ij_list, dim=0)
    logits_ji_list = torch.stack(logits_ji_list, dim=0)

    # calculate the loss
    loss_sdtw_ij = torch.mean(logits_ij_list)
    loss_sdtw_ji = torch.mean(logits_ji_list)

    if revcons:
        loss_con = classification_loss(logits, labels, label_smoothing)
        loss = loss_con + 0.1 * loss_sdtw_ij + 0.1 * loss_sdtw_ji
    else:
        loss = 0.1 * loss_sdtw_ij + 0.1 * loss_sdtw_ji
    return loss


def compute_dtw_alignment_contrastive_loss(embs,
                                           pos_indices,
                                           distance_type,
                                           softning,
                                           gamma_s,
                                           gamma_f,
                                           beta,
                                           loss_ratio,
                                           scale_factor,
                                           cyclic=False):
    # embs - [batch_size, 2, num_steps, dim]

    batch_size = embs.shape[0]
    logits_list = []
    loss_list = []
    for i in range(batch_size):
        embs1 = embs[i][0]
        embs2 = embs[i][1]
        logits, _ = smoothDTW(embs1, embs2, distance_type, softning, gamma_s, gamma_f)
        logits_list.append(scale_factor * logits[-1, -1])

        if cyclic:
            if random.random() < 0.5:  # cyclic action, align first half
                subset_indices1 = pos_indices[i, 0, 0:int(pos_indices.shape[-1] / 2)]
                subset_indices2 = pos_indices[i, 1, 0:int(pos_indices.shape[-1] / 2)]
            else:  # align second half
                subset_indices1 = pos_indices[i, 0, int(pos_indices.shape[-1] / 2):]
                subset_indices2 = pos_indices[i, 1, int(pos_indices.shape[-1] / 2):]
        else:
            subset_indices1 = pos_indices[i, 0, :]
            subset_indices2 = pos_indices[i, 1, :]

        embs1_ss = embs1[subset_indices1]
        embs2_ss = embs2[subset_indices2]

        logits1, _ = smoothDTW(embs1_ss, embs2_ss, distance_type, softning, gamma_s, gamma_f)
        embs2_ss = torch.flip(embs2_ss, dims=[0])
        logits2, _ = smoothDTW(embs1_ss, embs2_ss, distance_type, softning, gamma_s, gamma_f)
        a = scale_factor * logits1[-1, -1]
        b = scale_factor * logits2[-1, -1]
        loss_diff = max(a - b + beta, torch.zeros_like(a))

        logits_list.append(a)
        loss_list.append(loss_diff)

    loss1 = torch.mean(torch.stack(logits_list, dim=0))
    loss2 = torch.mean(torch.stack(loss_list, dim=0))
    loss = loss1 + loss_ratio * loss2
    # print(loss1, loss2, loss)
    return loss, loss_ratio * loss2


def compute_dtw_loss(args,
                     embs,
                     pos_indices,
                     alignment_type='dtw',
                     similarity_type='cosine',
                     label_smoothing=0.1,
                     softning='dtw_prob',
                     gamma_s=0.1,
                     gamma_f=0.1,
                     cyclic_action=False):
    """Computes DTW alignment loss between sequences of embeddings."""

    if alignment_type == 'dtw':
        loss = compute_dtw_alignment_loss(embs=embs,
                                          distance_type=similarity_type,
                                          softning=softning,
                                          gamma_s=gamma_s,
                                          gamma_f=gamma_f)
        return args.dtw_scale_factor * loss, torch.zeros_like(loss)

    elif alignment_type == 'dtw_consistency':
        loss = compute_dtw_alignment_consistency_loss(embs=embs,
                                                      distance_type=similarity_type,
                                                      softning=softning,
                                                      gamma_s=gamma_s,
                                                      gamma_f=gamma_f,
                                                      label_smoothing=label_smoothing,
                                                      revcons=True
                                                      )
        return args.dtw_scale_factor * loss, torch.zeros_like(loss)

    elif alignment_type == 'dtw_contrastive':
        loss, loss2 = compute_dtw_alignment_contrastive_loss(embs=embs,
                                                             pos_indices=pos_indices,
                                                             distance_type=similarity_type,
                                                             softning=softning,
                                                             gamma_s=gamma_s,
                                                             gamma_f=gamma_f,
                                                             beta=args.dtw_beta,
                                                             loss_ratio=args.dtw_ratio,
                                                             scale_factor=args.dtw_scale_factor,
                                                             cyclic=cyclic_action)
        return loss, loss2

    else:
        raise NotImplementedError
