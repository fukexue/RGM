import torch
import scipy.optimize as opt
import numpy as np


def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:k]
    return np.column_stack(np.unravel_index(idx, a.shape))


def hungarian(s: torch.Tensor, n1=None, n2=None, Inlier_src_pre=None, Inlier_ref_pre=None, th=0.5):
    """
    Solve optimal LAP permutation by hungarian algorithm.
    :param s: input 3d tensor (first dimension represents batch)
    :param n1: [num of objs in dim1] (against padding)
    :param n2: [num of objs in dim2] (against padding)
    :return: optimal permutation matrix
    """
    device = s.device
    batch_num = s.shape[0]

    perm_mat = s.cpu().detach().numpy() * -1
    for b in range(batch_num):
        n1b = perm_mat.shape[1] if n1 is None else n1[b]
        n2b = perm_mat.shape[2] if n2 is None else n2[b]
        if Inlier_src_pre is not None and Inlier_ref_pre is not None:
            perm_mat_bcopy = perm_mat[b].copy()
            Inlier_src_pre_np = Inlier_src_pre[b].cpu().detach().numpy()
            Inlier_ref_pre_np = Inlier_ref_pre[b].cpu().detach().numpy()
            row = np.where(Inlier_src_pre_np > th)[0]
            col = np.where(Inlier_ref_pre_np > th)[0]
            perm_mat_bcopy = perm_mat_bcopy[row, :]
            perm_mat_bcopy = perm_mat_bcopy[:, col]
            row_ind, col_ind = opt.linear_sum_assignment(perm_mat_bcopy)
            perm_mat[b] = np.zeros_like(perm_mat[b])
            for i,j in zip(row_ind, col_ind):
                perm_mat[b, row[i], col[j]] = 1
            # if top_K is not None:
            #     s_clone = s.clone()
            #     topk_ind = torch.topk(s_clone[b].view(1, -1), 10)
            #     row = (topk_ind[1] / s_clone[b].shape[1]).cpu().numpy()
            #     col = (topk_ind[1] % s_clone[b].shape[1]).cpu().numpy()
            #     perm_mat_1 = np.zeros_like(perm_mat[b])
            #     for i, j in zip(row, col):
            #         perm_mat_1[i, j] = 1
            #     perm_mat[b] = perm_mat[b]*perm_mat_1
        else:
            row_ind, col_ind = opt.linear_sum_assignment(perm_mat[b, :n1b, :n2b])
            perm_mat[b] = np.zeros_like(perm_mat[b])
            perm_mat[b, row_ind, col_ind] = 1
            # # 当外部未计算指派矩阵行和列和的情况下，使用如下计算方式。
            # perm_mat_bcopy = perm_mat[b].copy()
            # perm_mat_bcopyz = perm_mat[b].copy()*-1
            # row = np.where(np.max(perm_mat_bcopyz, axis=1) > th)[0]
            # col = np.where(np.max(perm_mat_bcopyz, axis=0) > th)[0]
            # perm_mat_bcopy = perm_mat_bcopy[row, :]
            # perm_mat_bcopy = perm_mat_bcopy[:, col]
            # row_ind, col_ind = opt.linear_sum_assignment(perm_mat_bcopy)
            # perm_mat[b] = np.zeros_like(perm_mat[b])
            # for i,j in zip(row_ind, col_ind):
            #     perm_mat[b, row[i], col[j]] = 1
    perm_mat = torch.from_numpy(perm_mat).to(device)

    return perm_mat
