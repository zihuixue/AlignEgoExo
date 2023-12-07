import math
import numpy as np
import torch


def vava_loss(X, Y, maxIter=20, lambda1=1.0, lambda2=0.1, virtual_distance=5.0, zeta=0.5, delta=2.0, global_step=None):
    """
    The vava loss term
    X: a N*d matrix, represent input N frames and each frame is encoded with d dimention vector
    Y: a M*d matrix, represent input M frames and each frame is encoded with d dimention vector
    maxIter: total number of iterations
    lambda1: the weight of the IDM regularization, default value: 1.0
    lambda2: the weight of the KL-divergence regularization, default value: 0.1
    virtual_distance: the threhold value to clip the distance for virtual frame, default value 5.0
    zeta: the theshold value for virtual frame, default value 0.5
    delta: the parameter of the prior Gaussian distribution, default value: 2.0
    """

    N = X.shape[0]
    M = Y.shape[0]

    # We normalize the hyper-parameters based on the sequence length, and we found it brings better performance
    lambda1 = lambda1 * (N + M)
    lambda2 = lambda2 * (N * M) / 4.0

    D_x_y = torch.mean((X.unsqueeze(1) - Y.unsqueeze(0)) ** 2, dim=2)
    min_index = torch.argmin(D_x_y, dim=1)
    min_index = min_index.cpu().numpy()
    min_index = min_index.astype(np.float32)

    # add one more value for virtual frame, which is the first one
    N += 1
    M += 1

    # GMM
    power = int(np.sqrt(global_step + 1.0))
    phi = 0.999 ** power
    phi = min(phi, 0.999)
    phi = max(phi, 0.001)

    P = np.zeros((N, M))
    mid_para = 1.0 / (N ** 2) + 1 / (M ** 2)
    mid_para = np.sqrt(mid_para)
    # pi = math.pi
    pi = torch.tensor(np.pi, dtype=torch.float32)
    threshold_value = 2.0 * virtual_distance / (N + M)
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # the distance to diagonal
            d_prior = torch.abs(torch.tensor(i / N - j / M))
            d_prior = d_prior / mid_para
            # the distance to the most similar matching for a giving i, adding extra 1 for virtual frame
            if i > 1:
                d_similarity = torch.abs(torch.tensor(j / M - (min_index[i - 2] + 1) / M))
            else:
                d_similarity = torch.abs(torch.tensor(j / M - 1.0 / M))
            d_similarity = d_similarity / mid_para
            p_consistency = torch.exp(-d_prior ** 2.0 / (2.0 * delta ** 2)) / (delta * torch.sqrt(2.0 * pi))
            p_optimal = torch.exp(-d_similarity ** 2.0 / (2.0 * delta ** 2)) / (delta * torch.sqrt(2.0 * pi))
            P[i - 1, j - 1] = phi * p_consistency + (1.0 - phi) * p_optimal
            # virtual frame prior value
            if (i == 1 or j == 1) and not (i == j):
                d = torch.tensor(threshold_value * 1.5 / mid_para)
                P[i - 1, j - 1] = torch.exp(-d ** 2.0 / (2.0 * delta ** 2)) / (delta * torch.sqrt(2.0 * pi))
    P = torch.tensor(P, dtype=torch.float32).to(X.device)

    S = torch.zeros((N, M))
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            s_consistency = torch.abs(torch.tensor(i / N - j / M))
            if i > 1:
                s_optimal = torch.abs(torch.tensor(j / M - (min_index[i - 2] + 1) / M))
            else:
                s_optimal = torch.abs(torch.tensor(j / M - 1.0 / M))
            s_consistency = lambda1 / (s_consistency ** 2 + 1.0)
            s_optimal = lambda1 / (s_optimal ** 2 + 1.0)
            S[i - 1, j - 1] = phi * s_consistency + (1.0 - phi) * s_optimal
            if (i == 1 or j == 1) and not (i == j):
                s = threshold_value * 1.5
                S[i - 1, j - 1] = lambda1 / (s ** 2 + 1.0)

    S = torch.tensor(S, dtype=torch.float32).to(X.device)
    XX = torch.sum(X * X, dim=1, keepdim=True)
    Y_transpose = Y.t()
    YY = torch.sum(Y_transpose * Y_transpose, dim=0, keepdim=True)
    XX = XX.repeat(1, M - 1)
    YY = YY.repeat(N - 1, 1)
    D = XX + YY - 2.0 * X @ Y_transpose
    bin1 = torch.full((1, M - 1), zeta).to(X.device)
    bin2 = torch.full((N, 1), zeta).to(X.device)
    D = torch.cat([bin1, D], dim=0)
    D = torch.cat([bin2, D], dim=1)

    K = P * torch.exp((S - D) / lambda2)
    K = torch.clamp(K, min=1e-15, max=1.0e20)

    a = (torch.ones([N, 1]) / N).to(X.device)
    b = (torch.ones([M, 1]) / M).to(X.device)

    ainvK = K / a
    compt = 0
    u = (torch.ones([N, 1]) / N).to(X.device)
    while compt < maxIter:
        Ktu = K.t() @ u
        aKtu = ainvK @ (b / Ktu)
        u = 1.0 / aKtu
        compt += 1

    new_Ktu = K.t() @ u
    v = b / new_Ktu

    aKv = ainvK @ v
    u = 1.0 / aKv

    U = K * D
    dis = torch.sum(u * (U @ v))
    dis = dis / (N * M * 1.0)
    return dis, U


def all_loss(X, Y, lambda3=2.0, delta=15.0, global_step=None, temperature=0.5):
    N = X.shape[0]
    M = Y.shape[0]
    assert X.shape[1] == Y.shape[1], 'The dimensions of instances in the input sequences must be the same!'

    W_x_p = torch.zeros((N, N)).to(X.device)
    for i in range(N):
        for j in range(N):
            W_x_p[i, j] = 1.0 / ((i - j) ** 2 + 1.0)

    y_x = torch.zeros((N, N)).to(X.device)
    for i in range(N):
        for j in range(N):
            if np.abs(i - j) > delta:
                y_x[i, j] = 1.0
            else:
                y_x[i, j] = 0.0

    W_y_p = torch.zeros((M, M)).to(X.device)
    for i in range(M):
        for j in range(M):
            W_y_p[i, j] = 1.0 / ((i - j) ** 2 + 1.0)

    y_y = torch.zeros((M, M)).to(X.device)
    for i in range(M):
        for j in range(M):
            if np.abs(i - j) > delta:
                y_y[i, j] = 1.0
            else:
                y_y[i, j] = 0.0

    D_x = torch.mean((X.unsqueeze(1) - X.unsqueeze(0)) ** 2, 2)
    D_y = torch.mean((Y.unsqueeze(1) - Y.unsqueeze(0)) ** 2, 2)

    C_x = torch.mean(y_x * torch.clamp(lambda3 - D_x, min=0.0) + (1.0 - y_x) * W_x_p * D_x)
    C_y = torch.mean(y_y * torch.clamp(lambda3 - D_y, min=0.0) + (1.0 - y_y) * W_y_p * D_y)

    vava_dis, U = vava_loss(X, Y, global_step=global_step)
    U = U[1:, 1:]
    X_best = torch.argmax(U, dim=1)
    X_worst = torch.argmin(U, dim=1)
    Y_best = torch.argmax(U, dim=0)
    Y_worst = torch.argmin(U, dim=0)

    best_distance = torch.mean((X - Y[X_best]) ** 2 + (Y - X[Y_best]) ** 2) / temperature
    worst_distance = torch.mean((X - Y[X_worst]) ** 2 + (Y - X[Y_worst]) ** 2) / temperature
    loss_inter = torch.nn.functional.cross_entropy(torch.tensor([0.0, 1.0]), torch.tensor([best_distance, worst_distance]))

    overall = 0.5 * (C_x + C_y) + vava_dis / (N * M) + 0.0001 * loss_inter
    return overall


def compute_vava_loss(embs, global_step):
    # embs - [batch_size, 2, num_steps, dim]
    batch_size = embs.shape[0]
    loss_list = []
    for i in range(batch_size):
        embs1 = embs[i][0]
        embs2 = embs[i][1]
        loss = all_loss(embs1, embs2, global_step=global_step)
        loss_list.append(loss)
    loss = torch.stack(loss_list, dim=0)
    return loss.mean()