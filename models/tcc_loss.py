import torch


def regression_loss(logits, labels, num_steps, steps, seq_lens, loss_type, normalize_indices, variance_lambda,
                    huber_delta):
    if normalize_indices:
        tile_seq_lens = seq_lens.unsqueeze(-1).repeat(1, num_steps)
        steps = steps / tile_seq_lens

    beta = torch.nn.functional.softmax(logits, dim=-1)

    true_time = (steps * labels).sum(dim=-1)
    pred_time = (steps * beta).sum(dim=-1)

    if loss_type in ['regression_mse', 'regression_mse_var']:
        if 'var' in loss_type:
            # Variance aware regression.
            pred_time_tiled = pred_time.unsqueeze(-1).repeat(1, num_steps)
            pred_time_variance = ((steps - pred_time_tiled) ** 2 * beta).sum(dim=-1)
            pred_time_log_var = torch.log(pred_time_variance)

            squared_error = (true_time - pred_time) ** 2

            loss = (torch.exp(-pred_time_log_var) * squared_error + variance_lambda * pred_time_log_var).mean()
            return loss
        else:
            import pdb
            pdb.set_trace()
    else:
        import pdb
        pdb.set_trace()

    return

    # pred_time_tiled = tf.tile(tf.expand_dims(pred_time, axis=1),
    #                             [1, num_steps])

    # pred_time_variance = tf.reduce_sum(
    #     tf.square(steps - pred_time_tiled) * beta, axis=1)


def align_pair_of_sequence(embs1, embs2, similarity_type, temperature):
    max_num_steps = embs1.shape[0]
    # Find distances between embs1 and embs2.
    sim_12 = get_scaled_similarity(embs1, embs2, similarity_type, temperature)
    # Softmax the distance.
    softmaxed_sim_12 = torch.nn.functional.softmax(sim_12, dim=1)

    # Calculate soft-nearest neighbors.
    # nn_embs = tf.matmul(softmaxed_sim_12, embs2)
    nn_embs = softmaxed_sim_12 @ embs2
    sim_21 = get_scaled_similarity(nn_embs, embs1, similarity_type, temperature)
    logits = sim_21

    labels = torch.arange(max_num_steps)
    labels_onehot = torch.nn.functional.one_hot(labels).to(logits)

    return logits, labels_onehot


def get_scaled_similarity(embs1, embs2, similarity_type, temperature):
    channels = embs1.shape[1]
    # Go for embs1 to embs2.
    if similarity_type == 'cosine':
        # similarity = tf.matmul(embs1, embs2, transpose_b=True)
        import pdb
        pdb.set_trace()
    elif similarity_type == 'l2':
        # embs1_ = embs1.unsqueeze(0)
        # embs2_ = embs2.unsqueeze(0)

        # similarity = torch.cdist(embs1_, embs2_).squeeze().contiguous()

        similarity = -1.0 * pairwise_l2_distance(embs1, embs2)
    else:
        raise ValueError('similarity_type can either be l2 or cosine.')

    # Scale the distance  by number of channels. This normalization helps with
    # optimization.
    similarity /= channels
    # Scale the distance by a temperature that helps with how soft/hard the
    # alignment should be.
    similarity /= temperature

    return similarity


def pairwise_l2_distance(embs1, embs2):
    """Computes pairwise distances between all rows of embs1 and embs2."""
    # norm1 = (embs1 ** 2).sum().reshape(-1, 1)
    # norm2 = (embs2 ** 2).sum().reshape(-1, 1)
    # dist = norm1 + norm2 - 2 * embs1 @ embs2.T

    norm1 = torch.sum(embs1 ** 2, dim=1, keepdim=True)
    norm2 = torch.sum(embs2 ** 2, dim=1, keepdim=True)
    dist = torch.mm(embs1, embs2.t()) * -2 + norm1 + norm2.t()
    dist = torch.clamp(dist, min=0.0)
    return dist


def compute_tcc_loss(embs, steps, seq_lens, temp):
    """
    :param embs: [batch_size, 2, num_steps, dim]
    :param steps: [batch_size, 2, num_steps]
    :param seq_lens: [batch_size, 2]
    :param temp: temperature
    :return: tcc loss
    """
    batch_size = embs.shape[0]
    num_steps = embs.shape[2]

    labels_list = []
    logits_list = []
    steps_list = []
    seq_lens_list = []

    for i in range(batch_size):
        embs1 = embs[i][0]
        embs2 = embs[i][1]
        logits, labels = align_pair_of_sequence(embs1, embs2, similarity_type="l2", temperature=temp)
        logits_list.append(logits)
        labels_list.append(labels)
        steps_list.append(steps[i][0].unsqueeze(0).repeat(num_steps, 1))
        seq_lens_list.append(seq_lens[i][0].repeat(num_steps))
        # steps_list.append(tf.tile(steps[i:i+1], [num_steps, 1]))
        # seq_lens_list.append(tf.tile(seq_lens[i:i+1], [num_steps]))

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    steps = torch.cat(steps_list, dim=0)
    seq_lens = torch.cat(seq_lens_list, dim=0)

    cycle_loss = regression_loss(logits, labels, num_steps, steps, seq_lens,
                                 loss_type="regression_mse_var", normalize_indices=True,
                                 variance_lambda=0.001, huber_delta=0.1)

    return cycle_loss
