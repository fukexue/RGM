#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import h5py
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.dcputil import quat2mat


_raw_features_sizes = {'xyz': 3, 'lxyz': 3, 'gxyz': 3, 'ppf': 4, 'pcf': 6}

# Part of the code is referred from: http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding

def nearest_neighbor(src, dst):
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)  # src, dst (num_dims, num_points)
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2,
                                                                                                           dim=0,
                                                                                                           keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances, indices


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def angle(v1: torch.Tensor, v2: torch.Tensor):
    """Compute angle between 2 vectors

    For robustness, we use the same formulation as in PPFNet, i.e.
        angle(v1, v2) = atan2(cross(v1, v2), dot(v1, v2)).
    This handles the case where one of the vectors is 0.0, since torch.atan2(0.0, 0.0)=0.0

    Args:
        v1: (B, *, 3)
        v2: (B, *, 3)

    Returns:

    """

    cross_prod = torch.stack([v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
                              v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
                              v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]], dim=-1)
    cross_prod_norm = torch.norm(cross_prod, dim=-1)
    dot_prod = torch.sum(v1 * v2, dim=-1)

    return torch.atan2(cross_prod_norm, dot_prod)


def get_graph_feature(data, feature_name, k=20):
    xyz = data[:, :3, :]
    # x = x.squeeze()
    idx = knn(xyz, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    # device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size).to(xyz.device).view(-1, 1, 1) * num_points
    feature = torch.tensor([]).to(xyz.device)

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = xyz.size()

    xyz = xyz.transpose(2, 1).contiguous()
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    #  batch_size * num_points * k + range(0, batch_size*num_points)

    # gxyz
    neighbor_gxyz = xyz.view(batch_size * num_points, -1)[idx, :]
    neighbor_gxyz = neighbor_gxyz.view(batch_size, num_points, k, num_dims)
    if 'gxyz' in feature_name:
        feature = torch.cat((feature, neighbor_gxyz), dim=3)

    # xyz
    xyz = xyz.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature, xyz), dim=3)


    # lxyz
    if 'lxyz' in feature_name:
        neighbor_lxyz = neighbor_gxyz - xyz
        feature = torch.cat((feature, neighbor_lxyz), dim=3)

    # ppf
    if 'ppf' in feature_name:
        normal = data[:, 3:6, :]
        normal = normal.transpose(2, 1).contiguous()
        neighbor_norm = normal.view(batch_size * num_points, -1)[idx, :]
        neighbor_norm = neighbor_norm.view(batch_size, num_points, k, num_dims)
        nr_d = angle(normal.permute(0, 2, 1).contiguous()[:,:,None,:], neighbor_lxyz)
        ni_d = angle(neighbor_norm, neighbor_lxyz)
        nr_ni = angle(normal.permute(0, 2, 1).contiguous()[:,:,None,:], neighbor_norm)
        d_norm = torch.norm(neighbor_lxyz, dim=-1)
        ppf_feat = torch.stack([nr_d, ni_d, nr_ni, d_norm], dim=-1)  # (B, npoint, n_sample, 4)
        feature = torch.cat((feature, ppf_feat), dim=3)

    # pcf
    if 'pcf' in feature_name:
        neighbor_gxyz_center = torch.mean(neighbor_gxyz, dim=2, keepdim=True)
        nrnc = neighbor_gxyz_center - xyz
        ncni = neighbor_gxyz - neighbor_gxyz_center
        ninr = xyz - neighbor_gxyz
        nrnc_norm = torch.norm(nrnc, dim=3)
        ncni_norm = torch.norm(ncni, dim=3)
        ninr_norm = torch.norm(ninr, dim=3)
        nr_angle = angle(nrnc, -ninr)
        nc_angle = angle(ncni, -nrnc)
        ni_angle = angle(ninr, -ncni)
        pcf_feat = torch.stack([nrnc_norm, ncni_norm, ninr_norm, nr_angle, nc_angle, ni_angle], dim=-1)
        feature = torch.cat((feature, pcf_feat), dim=3)

    feature = feature.permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNN(nn.Module):
    def __init__(self, features, neighboursnum, emb_dims=512):
        super(DGCNN, self).__init__()

        # 确定输入的点云信息
        self.features = features
        self.neighboursnum = neighboursnum
        raw_dim = sum([_raw_features_sizes[f] for f in self.features])  # number of channels after concat

        self.conv1 = nn.Conv2d(raw_dim, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, xyz):

        xyz = xyz.permute(0, 2, 1).contiguous()  # (B, 3, N)

        batch_size, num_dims, num_points = xyz.size()
        x = get_graph_feature(xyz, self.features, self.neighboursnum)   # (B, C, N, n)

        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x_node = x.squeeze(-1)

        x_edge = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        # if torch.sum(torch.isnan(x_edge)):
        #     print('discover nan value')
        return x_node, x_edge


class Classify1(nn.Module):
    def __init__(self, emb_dims=20):
        super(Classify, self).__init__()
        self.conv1 = nn.Conv1d(emb_dims, 256, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(128, 1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x, y):
        x = x.permute(0, 2, 1).contiguous()
        y = y.permute(0, 2, 1).contiguous()
        batch_size, _, num_points = x.size()
        x = get_graph_feature_cross(x, y)   #(B, K, N)
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))

        x_inlier = torch.sigmoid(self.conv3(x)).permute(0,2,1).contiguous()

        # x_edge = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        if torch.sum(torch.isnan(x_inlier)):
            print('discover nan value')
        return x_inlier


def knn_cross(x, y, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), y)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(2, 1).contiguous()

    dist = pairwise_distance.topk(k=k, dim=-1)[0]  # (batch_size, num_points, k)
    return dist


def get_graph_feature_cross(x, y, k=20):
    # x = x.squeeze()
    dist = knn_cross(x, y, k=k)  # (batch_size, num_points, k)
    dist = dist.permute(0, 2, 1).contiguous()
    return dist


class Classify2(nn.Module):
    def __init__(self, emb_dims=512):
        super(Classify2, self).__init__()
        self.conv00 = nn.Conv1d(emb_dims, emb_dims, kernel_size=1, bias=False)
        self.conv01 = nn.Conv1d(emb_dims, emb_dims, kernel_size=1, bias=False)
        self.bn00 = nn.BatchNorm1d(emb_dims)
        self.bn01 = nn.BatchNorm1d(emb_dims)
        self.conv1 = nn.Conv1d(emb_dims*2, 256, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(128, 1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x, y):
        # x = x.permute(0,2,1).contiguous()
        batch_size, _, num_points = x.size()
        # x = get_graph_feature(x)
        # y = y.permute(0,2,1).contiguous()
        y = F.relu(self.bn00(self.conv00(y)))
        y = F.relu(self.bn01(self.conv01(y)))
        y = torch.max(y, dim=2, keepdim=True)[0].repeat(1, 1, num_points)
        x = torch.cat((x,y), dim=1)

        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))

        x_inlier = torch.sigmoid(self.conv3(x)).permute(0,2,1).contiguous()

        # x_edge = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        if torch.sum(torch.isnan(x_inlier)):
            print('discover nan value')
        return x_inlier


class Classify(nn.Module):
    def __init__(self, emb_dims=512):
        super(Classify, self).__init__()
        self.conv1 = nn.Conv1d(emb_dims, 256, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(128, 1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.permute(0,2,1).contiguous()
        batch_size, _, num_points = x.size()
        # x = get_graph_feature(x)
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))

        x_inlier = torch.sigmoid(self.conv3(x)).permute(0,2,1).contiguous()

        # x_edge = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        if torch.sum(torch.isnan(x_inlier)):
            print('discover nan value')
        return x_inlier


class s_weight(nn.Module):
    def __init__(self, emb_dims=512):
        super(s_weight, self).__init__()
        self.prepool = nn.Sequential(
            nn.Conv1d(emb_dims+1, 1024, 1),
            nn.GroupNorm(16, 1024),
            nn.ReLU(),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.postpool = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.GroupNorm(16, 512),
            # nn.ReLU(),

            nn.Linear(512, 256),
            # nn.GroupNorm(16, 256),
            # nn.ReLU(),

            nn.Linear(256, 1),
        )

    def forward(self, src, tgt):
        src_padded = F.pad(src, (0, 1), mode='constant', value=0)
        ref_padded = F.pad(tgt, (0, 1), mode='constant', value=1)
        concatenated = torch.cat([src_padded, ref_padded], dim=1)

        prepool_feat = self.prepool(concatenated.permute(0, 2, 1))
        pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)
        raw_weights = self.postpool(pooled)

        # beta = torch.exp(10*raw_weights[:, 0])
        beta = F.softplus(raw_weights[:, 0])

        return beta