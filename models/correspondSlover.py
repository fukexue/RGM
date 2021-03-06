#!/usr/bin/env python
# -*- coding: utf-8 -*-


import math
import torch
import torch.nn as nn
from utils.dcputil import quat2mat
import open3d


class MLPHead(nn.Module):
    def __init__(self, args):
        super(MLPHead, self).__init__()
        emb_dims = args.emb_dims
        self.emb_dims = emb_dims
        self.nn = nn.Sequential(nn.Linear(emb_dims * 2, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        embedding = torch.cat((src_embedding, tgt_embedding), dim=1)
        embedding = self.nn(embedding.max(dim=-1)[0])
        rotation = self.proj_rot(embedding)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        translation = self.proj_trans(embedding)
        return quat2mat(rotation), translation


# class SVDHead(nn.Module):
#     def __init__(self):
#         super(SVDHead, self).__init__()
#         self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
#         self.reflect[2, 2] = -1

def SVDslover(src_o, tgt_o, s_perm_mat):
    # for i in range(s_perm_mat.shape[0]):
    #     pmat_predi_index1 = torch.where(s_perm_mat[i])
    #     if pmat_predi_index1[0].shape[0] != s_perm_mat[i].shape[0] and \
    #             pmat_predi_index1[1].shape[0] != s_perm_mat[i].shape[1]:
    #         src_in = src_o[i, pmat_predi_index1[0]]
    #         tgt_in = tgt_o[i, pmat_predi_index1[0]]
    #         src_cen = src_in.mean(dim=-2, keepdim=True)
    #         tgt_cen = tgt_in.mean(dim=-2, keepdim=True)
    #         pmat_predi_index0 = torch.where(torch.sum(s_perm_mat[i], dim=1)==0)
    #         tgt_o[i, pmat_predi_index0[0]] = src_cen
    #         tgt_o[i, pmat_predi_index0[0]] = tgt_cen
    # reflect = torch.eye(3).cuda()
    # reflect[2, 2] = -1
    # src = src_o.permute(0, 2, 1).contiguous()
    # tgt = tgt_o.permute(0, 2, 1).contiguous()
    # batch_size = src.size(0)
    #
    # src_centered = src - src.mean(dim=2, keepdim=True)
    #
    # tgt_centered = tgt - tgt.mean(dim=2, keepdim=True)
    #
    # H = torch.matmul(src_centered, tgt_centered.transpose(2, 1).contiguous())
    #
    # # U, S, V = [], [], []
    # R = []
    #
    # for i in range(src.size(0)):
    #     u, s, v = torch.svd(H[i])
    #     r = torch.matmul(v, u.transpose(1, 0).contiguous())
    #     r_det = torch.det(r)
    #     if r_det < 0:
    #         u, s, v = torch.svd(H[i])
    #         v = torch.matmul(v, reflect)
    #         r = torch.matmul(v, u.transpose(1, 0).contiguous())
    #         # r = r * self.reflect
    #     R.append(r)
    #
    # R = torch.stack(R, dim=0)
    #
    # t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + tgt.mean(dim=2, keepdim=True)

    """Compute rigid transforms between two point sets

    Args:
        src_o (torch.Tensor): (B, M, 3) points
        tgt_o (torch.Tensor): (B, N, 3) points
        s_perm_mat (torch.Tensor): (B, M, N)

    Returns:
        Transform R (B, 3, 3) t(B, 3) to get from src_o to tgt_o, i.e. T*src = tgt
    """
    weights = torch.sum(s_perm_mat, dim=2)
    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + 1e-5)
    centroid_src_o = torch.sum(src_o * weights_normalized, dim=1)
    centroid_tgt_o = torch.sum(tgt_o * weights_normalized, dim=1)
    src_o_centered = src_o - centroid_src_o[:, None, :]
    tgt_o_centered = tgt_o - centroid_tgt_o[:, None, :]
    cov = src_o_centered.transpose(-2, -1) @ (tgt_o_centered * weights_normalized)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    R = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(R) > 0)

    # Compute translation (uncenter centroid)
    t = -R @ centroid_src_o[:, :, None] + centroid_tgt_o[:, :, None]

    return R, t.view(s_perm_mat.shape[0], 3)


def RANSACSVDslover(src_o, tgt_o, s_perm_mat):
    """Compute rigid transforms between two point sets with RANSAC

    Args:
        src_o (torch.Tensor): (B, M, 3) points
        tgt_o (torch.Tensor): (B, N, 3) points
        s_perm_mat (torch.Tensor): (B, M, N)

    Returns:
        Transform R (B, 3, 3) t(B, 3) to get from src_o to tgt_o, i.e. T*src = tgt
    """
    weights = torch.sum(s_perm_mat, dim=2)
    weights_inlier = torch.where(weights==1)
    import numpy as np

    src_o0 = [src_o[i, list((weights_inlier[1][weights_inlier[0]==i]).cpu().numpy())].cpu().numpy()
              for i in range(s_perm_mat.shape[0])]
    tgt_o0 = [tgt_o[i, list((weights_inlier[1][weights_inlier[0]==i]).cpu().numpy())].cpu().numpy()
              for i in range(s_perm_mat.shape[0])]
    R = torch.zeros((s_perm_mat.shape[0], 3, 3)).to(s_perm_mat)
    t = torch.zeros((s_perm_mat.shape[0], 3, 1)).to(s_perm_mat)
    s_perm_mat_re = torch.zeros_like(s_perm_mat).to(s_perm_mat)
    for i in range(len(src_o0)):
        src_o0i = open3d.geometry.PointCloud()
        tgt_o0i = open3d.geometry.PointCloud()
        src_o0i.points = open3d.utility.Vector3dVector(src_o0[i])
        tgt_o0i.points = open3d.utility.Vector3dVector(tgt_o0[i])
        corr = open3d.utility.Vector2iVector(np.arange(src_o0[i].shape[0])[:,None].repeat(2, axis=1))
        reg_result = open3d.registration.registration_ransac_based_on_correspondence(src_o0i, tgt_o0i, corr, 0.2)
        R[i] = torch.from_numpy(reg_result.transformation[:3, :3]).to(s_perm_mat)
        t[i] = torch.from_numpy(reg_result.transformation[:3, 3])[:,None].to(s_perm_mat)
        corr_re = np.asarray(reg_result.correspondence_set)
        s_perm_mat_re[i,corr_re[:,0]] = s_perm_mat[i,corr_re[:,0]]

    return R, t.view(s_perm_mat.shape[0], 3), s_perm_mat_re