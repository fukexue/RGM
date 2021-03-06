import torch
import torch.nn as nn
import math

from utils.config import cfg
from models.dgcnn import DGCNN
from models.gconv import Siamese_Gconv
from models.affinity_layer import Affinity
from models.transformer import Transformer


def match_features(feat_src, feat_ref, metric='l2'):
    """ Compute pairwise distance between features

    Args:
        feat_src: (B, J, C)
        feat_ref: (B, K, C)
        metric: either 'angle' or 'l2' (squared euclidean)

    Returns:
        Matching matrix (B, J, K). i'th row describes how well the i'th point
         in the src agrees with every point in the ref.
    """
    assert feat_src.shape[-1] == feat_ref.shape[-1]

    if metric == 'l2':
        dist_matrix = square_distance(feat_src, feat_ref)
    else:
        raise NotImplementedError

    return dist_matrix


def compute_affinity(feat_distance, alpha=0.5):
    """Compute logarithm of Initial match matrix values, i.e. log(m_jk)"""
    hybrid_affinity = -1 * (feat_distance - alpha)
    return hybrid_affinity


def sinkhorn_rpm(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha


class GMAttend(nn.Module):
    def __init__(self, hidden_dim: int):
        super(GMAttend, self).__init__()
        self.key_dim = hidden_dim // 8
        self.query_w = nn.Linear(hidden_dim, self.key_dim)
        self.key_w = nn.Linear(hidden_dim, self.key_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        y_avg = torch.mean(y, dim=-2, keepdim=True).repeat(1, y.shape[1], 1)
        x = torch.cat([x, y_avg], dim=-1)
        queries = self.query_w(x)
        keys = self.key_w(x)
        attention = self.softmax(torch.einsum('bqf,bkf->bqk', queries, keys))
        return attention


class GMAttendSigmod(nn.Module):
    def __init__(self, hidden_dim: int):
        super(GMAttendSigmod, self).__init__()
        self.key_dim = hidden_dim // 8
        self.query_w = nn.Linear(hidden_dim, self.key_dim)
        self.key_w = nn.Linear(hidden_dim, self.key_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        y_avg = torch.mean(y, dim=-2, keepdim=True).repeat(1, y.shape[1], 1)
        x = torch.cat([x, y_avg], dim=-1)
        queries = self.query_w(x)
        keys = self.key_w(x)
        attention = self.sigmoid(torch.einsum('bqf,bkf->bqk', queries, keys))
        return attention


class GMAttendSigmod1(nn.Module):
    def __init__(self, hidden_dim: int):
        super(GMAttendSigmod1, self).__init__()
        self.key_dim = hidden_dim // 8
        self.query_w = nn.Linear(hidden_dim, self.key_dim)
        self.key_w = nn.Linear(hidden_dim, self.key_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        y_avg = torch.mean(y, dim=-2, keepdim=True).repeat(1, y.shape[1], 1)
        x = torch.cat([x, y_avg], dim=-1)
        queries = self.query_w(x)
        keys = self.key_w(x)
        atweight = torch.mean(self.sigmoid(torch.einsum('bqf,bkf->bqk', queries, keys)), dim=-1, keepdim=True)
        attention = torch.bmm(atweight, atweight.transpose(2,1))
        return attention


class GMAttendSigmod2(nn.Module):
    def __init__(self, hidden_dim: int):
        super(GMAttendSigmod2, self).__init__()
        self.key_dim = hidden_dim // 8
        self.query_w = nn.Linear(hidden_dim, self.key_dim)
        self.key_w = nn.Linear(hidden_dim, self.key_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        queries = self.query_w(x)
        keys = self.key_w(y)
        atweight = torch.max(self.sigmoid(torch.einsum('bqf,bkf->bqk', queries, keys)), dim=-1, keepdim=True).values
        attention = torch.bmm(atweight, atweight.transpose(2,1))
        return attention


class GMAttendSoftmax1(nn.Module):
    def __init__(self, hidden_dim: int):
        super(GMAttendSoftmax1, self).__init__()
        self.key_dim = hidden_dim // 8
        self.query_w = nn.Linear(hidden_dim, self.key_dim)
        self.key_w = nn.Linear(hidden_dim, self.key_dim)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x, y):
        queries = self.query_w(x)
        keys = self.key_w(y)
        atweight = torch.max(self.softmax(torch.einsum('bqf,bkf->bqk', queries, keys)), dim=-1, keepdim=True).values
        attention = torch.bmm(atweight, atweight.transpose(2,1))
        return attention


def square_distance(src, dst):
    """Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
             = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1)[:, :, None]
    dist += torch.sum(dst ** 2, dim=-1)[:, None, :]
    return dist


class GMAttendSoftmax2(nn.Module):
    def __init__(self, hidden_dim: int):
        super(GMAttendSoftmax2, self).__init__()
        self.key_dim = hidden_dim // 8
        self.query_w = nn.Linear(hidden_dim, self.key_dim)
        self.key_w = nn.Linear(hidden_dim, self.key_dim)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x, y):
        queries = self.query_w(x)
        keys = self.key_w(y)
        dist = square_distance(queries, keys)
        atweight = torch.max(torch.exp(-1*dist), dim=-1, keepdim=True).values
        attention = torch.bmm(atweight, atweight.transpose(2,1))
        return attention


class GMAttendSoftmax3(nn.Module):
    def __init__(self, hidden_dim: int):
        super(GMAttendSoftmax3, self).__init__()
        self.key_dim = hidden_dim // 8
        self.query_w = nn.Linear(hidden_dim, self.key_dim)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x, y):
        x_w = self.query_w(y)
        y_w = self.query_w(y)
        dist = square_distance(x_w, y_w)
        atweight = torch.max(torch.exp(-1*dist), dim=-1, keepdim=True).values
        attention = torch.bmm(atweight, atweight.transpose(2,1))
        return attention


class GMAttendSoftmax4(nn.Module):
    def __init__(self, hidden_dim: int):
        super(GMAttendSoftmax4, self).__init__()
        self.key_dim = hidden_dim // 8
        self.query_w = nn.Linear(hidden_dim, self.key_dim)
        self.instnorm = nn.InstanceNorm2d(1, affine=True)

    def forward(self, x, y):
        x_w = self.query_w(x)
        y_w = self.query_w(y)
        dist = square_distance(x_w, y_w)
        dist_s = self.instnorm(torch.exp(-1*dist)[:, None, :, :]).squeeze(dim=1)
        log_dist_s = sinkhorn_rpm(dist_s, n_iters=20, slack=cfg.PGM.SKADDCR)
        dist_s = torch.exp(log_dist_s)
        atweight = torch.max(dist_s, dim=-1, keepdim=True).values
        attention = torch.bmm(atweight, atweight.transpose(2,1))
        return attention


class GMAttendattention(nn.Module):
    def __init__(self, hidden_dim: int):
        super(GMAttendattention, self).__init__()
        self.key_dim = hidden_dim // 8
        self.query_w = nn.Linear(hidden_dim, self.key_dim)
        self.key_w = nn.Linear(hidden_dim, self.key_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.query_w1 = nn.Linear(hidden_dim//2, self.key_dim)
        self.key_w1 = nn.Linear(hidden_dim//2, self.key_dim)
        self.softmax1 = nn.Softmax(dim=-1)

    def forward(self, x, y):
        queries1 = self.query_w1(y)
        keys1 = self.key_w1(y)
        attention1 = self.softmax1(torch.sum(torch.einsum('bqf,bkf->bqk', queries1, keys1), dim=-2, keepdim=True))
        y_avg = torch.bmm(attention1, y)
        y_avg = y_avg.repeat(1, y.shape[1], 1)
        x = torch.cat([x, y_avg], dim=-1)
        queries = self.query_w(x)
        keys = self.key_w(x)
        attention = self.softmax(torch.einsum('bqf,bkf->bqk', queries, keys))
        return attention


class GMAttendMaxpool(nn.Module):
    def __init__(self, hidden_dim: int):
        super(GMAttendMaxpool, self).__init__()
        self.key_dim = hidden_dim // 8
        self.query_w = nn.Linear(hidden_dim, self.key_dim)
        self.key_w = nn.Linear(hidden_dim, self.key_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        y_max = y.max(dim=-2, keepdim=True)[0].repeat(1, y.shape[1], 1)
        x = torch.cat([x, y_max], dim=-1)
        queries = self.query_w(x)
        keys = self.key_w(x)
        attention = self.softmax(torch.einsum('bqf,bkf->bqk', queries, keys))
        return attention


class GMAttendNother(nn.Module):
    def __init__(self, hidden_dim: int):
        super(GMAttendNother, self).__init__()
        self.key_dim = hidden_dim // 8
        self.query_w = nn.Linear(hidden_dim, self.key_dim)
        self.key_w = nn.Linear(hidden_dim, self.key_dim)
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x):
        x = x[:,None,:,:]
        queries = self.query_w(x)
        keys = self.key_w(x)
        attention = self.softmax(torch.einsum('bqf,bkf->bqk', queries, keys))
        return attention


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pointfeaturer = DGCNN(cfg.PGM.FEATURES, cfg.PGM.NEIGHBORSNUM, cfg.PGM.FEATURE_EDGE_CHANNEL)

        # self.bi_stochastic = Sinkhorn(max_iter=cfg.PGM.BS_ITER_NUM, epsilon=cfg.PGM.BS_EPSILON,
        #                               skaddcr=cfg.PGM.SKADDCR, skaddcr_value=cfg.PGM.SKADDCRVALUE)
        # self.voting_layer = Voting(alpha=cfg.PGM.VOTING_ALPHA)
        # self.displacement_layer = Displacement()
        # self.l2norm = nn.LocalResponseNorm(cfg.PGM.FEATURE_EDGE_CHANNEL*2,
        #                                    alpha=cfg.PGM.FEATURE_EDGE_CHANNEL*2, beta=0.5, k=0)
        self.gnn_layer = cfg.PGM.GNN_LAYER
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(cfg.PGM.FEATURE_NODE_CHANNEL + cfg.PGM.FEATURE_EDGE_CHANNEL, cfg.PGM.GNN_FEAT)
            else:
                gnn_layer = Siamese_Gconv(cfg.PGM.GNN_FEAT, cfg.PGM.GNN_FEAT)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(cfg.PGM.GNN_FEAT))
            if cfg.PGM.USEATTEND == 'attentionsplit':
                self.add_module('gmattend_st{}'.format(i), GMAttend(4*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                    if i==0 else 2*cfg.PGM.GNN_FEAT))
                self.add_module('gmattend_ts{}'.format(i), GMAttend(4*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                    if i==0 else 2*cfg.PGM.GNN_FEAT))
            elif cfg.PGM.USEATTEND == 'attentionsplitinlier':
                if i==0 :
                    self.inlierfeaturer = DGCNN(cfg.PGM.FEATURES, cfg.PGM.NEIGHBORSNUM, cfg.PGM.FEATURE_EDGE_CHANNEL)
                self.add_module('gmattend_st{}'.format(i), GMAttend(4*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                    if i==0 else 2*cfg.PGM.GNN_FEAT))
                self.add_module('gmattend_ts{}'.format(i), GMAttend(4*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                    if i==0 else 2*cfg.PGM.GNN_FEAT))
            elif cfg.PGM.USEATTEND == 'attentiontransformernoais':
                self.add_module('gmattend{}'.format(i), Transformer(2 * cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                    if i == 0 else cfg.PGM.GNN_FEAT))
            elif cfg.PGM.USEATTEND == 'attentiontransformersplit':
                self.add_module('gmattend_st{}'.format(i), Transformer(2 * cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                       if i == 0 else cfg.PGM.GNN_FEAT))
                self.add_module('gmattend_ts{}'.format(i), Transformer(2 * cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                       if i == 0 else cfg.PGM.GNN_FEAT))
            elif cfg.PGM.USEATTEND == 'attentionplussplit':
                self.add_module('gmattend_st{}'.format(i), GMAttendattention(4*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                             if i==0 else 2*cfg.PGM.GNN_FEAT))
                self.add_module('gmattend_ts{}'.format(i), GMAttendattention(4*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                             if i==0 else 2*cfg.PGM.GNN_FEAT))
            elif cfg.PGM.USEATTEND == 'attentionsigmoid':
                self.add_module('gmattend_st{}'.format(i), GMAttendSigmod(4*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                          if i==0 else 2*cfg.PGM.GNN_FEAT))
                self.add_module('gmattend_ts{}'.format(i), GMAttendSigmod(4*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                          if i==0 else 2*cfg.PGM.GNN_FEAT))
            elif cfg.PGM.USEATTEND == 'attentionsigmoid1':
                self.add_module('gmattend_st{}'.format(i), GMAttendSigmod1(4*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                          if i==0 else 2*cfg.PGM.GNN_FEAT))
                self.add_module('gmattend_ts{}'.format(i), GMAttendSigmod1(4*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                          if i==0 else 2*cfg.PGM.GNN_FEAT))
            elif cfg.PGM.USEATTEND == 'attentionsigmoid2':
                self.add_module('gmattend_st{}'.format(i), GMAttendSigmod2(2*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                          if i==0 else cfg.PGM.GNN_FEAT))
                self.add_module('gmattend_ts{}'.format(i), GMAttendSigmod2(2*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                          if i==0 else cfg.PGM.GNN_FEAT))
            elif cfg.PGM.USEATTEND == 'attentionsoftmax1':
                self.add_module('gmattend_st{}'.format(i), GMAttendSoftmax1(2*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                            if i==0 else cfg.PGM.GNN_FEAT))
                self.add_module('gmattend_ts{}'.format(i), GMAttendSoftmax1(2*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                            if i==0 else cfg.PGM.GNN_FEAT))
            elif cfg.PGM.USEATTEND == 'attentionsoftmax2':
                self.add_module('gmattend_st{}'.format(i), GMAttendSoftmax2(2*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                            if i==0 else cfg.PGM.GNN_FEAT))
                self.add_module('gmattend_ts{}'.format(i), GMAttendSoftmax2(2*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                            if i==0 else cfg.PGM.GNN_FEAT))
            elif cfg.PGM.USEATTEND == 'attentionsoftmax3':
                if i==0 :
                    self.inlierfeaturer = DGCNN(cfg.PGM.FEATURES, cfg.PGM.NEIGHBORSNUM, cfg.PGM.FEATURE_EDGE_CHANNEL)
                self.add_module('gmattend_st{}'.format(i), GMAttendSoftmax3(2*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                            if i==0 else cfg.PGM.GNN_FEAT))
                self.add_module('gmattend_ts{}'.format(i), GMAttendSoftmax3(2*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                            if i==0 else cfg.PGM.GNN_FEAT))
            elif cfg.PGM.USEATTEND == 'attentionsoftmax4':
                if i==0 :
                    self.inlierfeaturer = DGCNN(cfg.PGM.FEATURES, cfg.PGM.NEIGHBORSNUM, cfg.PGM.FEATURE_EDGE_CHANNEL)
                self.add_module('gmattend_st{}'.format(i), GMAttendSoftmax4(2*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                            if i==0 else cfg.PGM.GNN_FEAT))
                self.add_module('gmattend_ts{}'.format(i), GMAttendSoftmax4(2*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                            if i==0 else cfg.PGM.GNN_FEAT))
            elif cfg.PGM.USEATTEND == 'attentionshare':
                self.add_module('gmattend{}'.format(i), GMAttend(4*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                 if i==0 else 2*cfg.PGM.GNN_FEAT))
            elif cfg.PGM.USEATTEND == 'attentionsplitNoother':
                self.add_module('gmattend_st{}'.format(i), GMAttendNother(2*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                          if i==0 else cfg.PGM.GNN_FEAT))
                self.add_module('gmattend_ts{}'.format(i), GMAttendNother(2*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                          if i==0 else cfg.PGM.GNN_FEAT))
            elif cfg.PGM.USEATTEND == 'attentionshareNoother':
                self.add_module('gmattend{}'.format(i), GMAttendNother(2*cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                       if i==0 else cfg.PGM.GNN_FEAT))
            self.add_module('InstNorm_layer_{}'.format(i), nn.InstanceNorm2d(1, affine=True))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), nn.Linear(cfg.PGM.GNN_FEAT * 2, cfg.PGM.GNN_FEAT))

        # if cfg.DATASET.NOISE_TYPE == 'clean':
        #     self.classify_layer = None
        # else:
        #     # self.classify_layer = Classify()
        #     self.classify_layer = True

    def forward(self, P_src, P_tgt, A_src, A_tgt, ns_src, ns_tgt):
        # extract feature
        Node_src, Edge_src = self.pointfeaturer(P_src)
        Node_tgt, Edge_tgt = self.pointfeaturer(P_tgt)

        emb_src, emb_tgt = torch.cat((Node_src, Edge_src), dim=1).transpose(1, 2).contiguous(), \
                           torch.cat((Node_tgt, Edge_tgt), dim=1).transpose(1, 2).contiguous()
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            if cfg.PGM.USEATTEND == 'attentionsplit':
                gmattends_layer = getattr(self, 'gmattend_st{}'.format(i))
                gmattendt_layer = getattr(self, 'gmattend_ts{}'.format(i))
                A_src1 = gmattends_layer(emb_src, emb_tgt)
                A_tgt1 = gmattendt_layer(emb_tgt, emb_src)
                emb_src, emb_tgt = gnn_layer([A_src1, emb_src], [A_tgt1, emb_tgt])
            elif cfg.PGM.USEATTEND == 'attentionsplitinlier':
                gmattends_layer = getattr(self, 'gmattend_st{}'.format(i))
                gmattendt_layer = getattr(self, 'gmattend_ts{}'.format(i))
                if i==0:
                    Node_src1, Edge_src1 = self.inlierfeaturer(P_src)
                    Node_tgt1, Edge_tgt1 = self.inlierfeaturer(P_tgt)
                    emb_src1, emb_tgt1 = torch.cat((Node_src1, Edge_src1), dim=1).transpose(1, 2).contiguous(), \
                                         torch.cat((Node_tgt1, Edge_tgt1), dim=1).transpose(1, 2).contiguous()
                    A_src1 = gmattends_layer(emb_src1, emb_tgt1)
                    A_tgt1 = gmattendt_layer(emb_tgt1, emb_src1)
                else:
                    A_src1 = gmattends_layer(emb_src, emb_tgt)
                    A_tgt1 = gmattendt_layer(emb_tgt, emb_src)
                emb_src, emb_tgt = gnn_layer([A_src1, emb_src], [A_tgt1, emb_tgt])
            elif cfg.PGM.USEATTEND == 'attentiontransformernoais':
                gmattends_layer = getattr(self, 'gmattend{}'.format(i))
                src_embedding, tgt_embedding = gmattends_layer(emb_src, emb_tgt)
                d_k = src_embedding.size(1)
                scores_src = torch.matmul(src_embedding.transpose(2, 1).contiguous(), src_embedding) / math.sqrt(d_k)
                scores_tgt = torch.matmul(tgt_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
                A_src1 = torch.softmax(scores_src, dim=-1)
                A_tgt1 = torch.softmax(scores_tgt, dim=-1)
                emb_src, emb_tgt = gnn_layer([A_src1, emb_src], [A_tgt1, emb_tgt])
            elif cfg.PGM.USEATTEND == 'attentiontransformersplit':
                gmattends_layer = getattr(self, 'gmattend_st{}'.format(i))
                gmattendt_layer = getattr(self, 'gmattend_ts{}'.format(i))
                src_embedding, _ = gmattends_layer(emb_src, emb_tgt)
                tgt_embedding, _ = gmattendt_layer(emb_tgt, emb_src)
                d_ksrc = src_embedding.size(1)
                d_ktgt = tgt_embedding.size(1)
                scores_src = torch.matmul(src_embedding.transpose(2, 1).contiguous(), src_embedding) / math.sqrt(d_ksrc)
                scores_tgt = torch.matmul(tgt_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_ktgt)
                A_src1 = torch.softmax(scores_src, dim=-1)
                A_tgt1 = torch.softmax(scores_tgt, dim=-1)
                emb_src, emb_tgt = gnn_layer([A_src1, emb_src], [A_tgt1, emb_tgt])
            elif cfg.PGM.USEATTEND == 'attentionplussplit':
                gmattends_layer = getattr(self, 'gmattend_st{}'.format(i))
                gmattendt_layer = getattr(self, 'gmattend_ts{}'.format(i))
                A_src1 = gmattends_layer(emb_src, emb_tgt)
                A_tgt1 = gmattendt_layer(emb_tgt, emb_src)
                emb_src, emb_tgt = gnn_layer([A_src1, emb_src], [A_tgt1, emb_tgt])
            elif cfg.PGM.USEATTEND == 'attentionsigmoid':
                gmattends_layer = getattr(self, 'gmattend_st{}'.format(i))
                gmattendt_layer = getattr(self, 'gmattend_ts{}'.format(i))
                A_src1 = gmattends_layer(emb_src, emb_tgt)
                A_tgt1 = gmattendt_layer(emb_tgt, emb_src)
                emb_src, emb_tgt = gnn_layer([A_src1, emb_src], [A_tgt1, emb_tgt])
            elif cfg.PGM.USEATTEND == 'attentionsigmoid1':
                gmattends_layer = getattr(self, 'gmattend_st{}'.format(i))
                gmattendt_layer = getattr(self, 'gmattend_ts{}'.format(i))
                A_src1 = gmattends_layer(emb_src, emb_tgt)
                A_tgt1 = gmattendt_layer(emb_tgt, emb_src)
                emb_src, emb_tgt = gnn_layer([A_src1, emb_src], [A_tgt1, emb_tgt])
            elif cfg.PGM.USEATTEND == 'attentionsigmoid2':
                gmattends_layer = getattr(self, 'gmattend_st{}'.format(i))
                gmattendt_layer = getattr(self, 'gmattend_ts{}'.format(i))
                A_src1 = gmattends_layer(emb_src, emb_tgt)
                A_tgt1 = gmattendt_layer(emb_tgt, emb_src)
                emb_src, emb_tgt = gnn_layer([A_src1, emb_src], [A_tgt1, emb_tgt])
            elif cfg.PGM.USEATTEND == 'attentionsoftmax1':
                gmattends_layer = getattr(self, 'gmattend_st{}'.format(i))
                gmattendt_layer = getattr(self, 'gmattend_ts{}'.format(i))
                A_src1 = gmattends_layer(emb_src, emb_tgt)
                A_tgt1 = gmattendt_layer(emb_tgt, emb_src)
                emb_src, emb_tgt = gnn_layer([A_src1, emb_src], [A_tgt1, emb_tgt])
            elif cfg.PGM.USEATTEND == 'attentionsoftmax2':
                gmattends_layer = getattr(self, 'gmattend_st{}'.format(i))
                gmattendt_layer = getattr(self, 'gmattend_ts{}'.format(i))
                A_src1 = gmattends_layer(emb_src, emb_tgt)
                A_tgt1 = gmattendt_layer(emb_tgt, emb_src)
                emb_src, emb_tgt = gnn_layer([A_src1, emb_src], [A_tgt1, emb_tgt])
            elif cfg.PGM.USEATTEND == 'attentionsoftmax3':
                gmattends_layer = getattr(self, 'gmattend_st{}'.format(i))
                gmattendt_layer = getattr(self, 'gmattend_ts{}'.format(i))
                if i==0:
                    Node_src1, Edge_src1 = self.inlierfeaturer(P_src)
                    Node_tgt1, Edge_tgt1 = self.inlierfeaturer(P_tgt)
                    emb_src1, emb_tgt1 = torch.cat((Node_src1, Edge_src1), dim=1).transpose(1, 2).contiguous(), \
                                         torch.cat((Node_tgt1, Edge_tgt1), dim=1).transpose(1, 2).contiguous()
                    A_src1 = gmattends_layer(emb_src1, emb_tgt1)
                    A_tgt1 = gmattendt_layer(emb_tgt1, emb_src1)
                else:
                    A_src1 = gmattends_layer(emb_src, emb_tgt)
                    A_tgt1 = gmattendt_layer(emb_tgt, emb_src)
                emb_src, emb_tgt = gnn_layer([A_src1, emb_src], [A_tgt1, emb_tgt])
            elif cfg.PGM.USEATTEND == 'attentionsoftmax4':
                gmattends_layer = getattr(self, 'gmattend_st{}'.format(i))
                gmattendt_layer = getattr(self, 'gmattend_ts{}'.format(i))
                if i==0:
                    Node_src1, Edge_src1 = self.inlierfeaturer(P_src)
                    Node_tgt1, Edge_tgt1 = self.inlierfeaturer(P_tgt)
                    emb_src1, emb_tgt1 = torch.cat((Node_src1, Edge_src1), dim=1).transpose(1, 2).contiguous(), \
                                         torch.cat((Node_tgt1, Edge_tgt1), dim=1).transpose(1, 2).contiguous()
                    A_src1 = gmattends_layer(emb_src1, emb_tgt1)
                    A_tgt1 = gmattendt_layer(emb_tgt1, emb_src1)
                else:
                    A_src1 = gmattends_layer(emb_src, emb_tgt)
                    A_tgt1 = gmattendt_layer(emb_tgt, emb_src)
                emb_src, emb_tgt = gnn_layer([A_src1, emb_src], [A_tgt1, emb_tgt])
            elif cfg.PGM.USEATTEND == 'attentionshare':
                gmattend_layer = getattr(self, 'gmattend{}'.format(i))
                A_src1 = gmattend_layer(emb_src, emb_tgt)
                A_tgt1 = gmattend_layer(emb_tgt, emb_src)
                emb_src, emb_tgt = gnn_layer([A_src1, emb_src], [A_tgt1, emb_tgt])
            elif cfg.PGM.USEATTEND == 'attentionsplitNoother':
                gmattends_layer = getattr(self, 'gmattend_st{}'.format(i))
                gmattendt_layer = getattr(self, 'gmattend_ts{}'.format(i))
                A_src1 = gmattends_layer(emb_src)
                A_tgt1 = gmattendt_layer(emb_tgt)
                emb_src, emb_tgt = gnn_layer([A_src1, emb_src], [A_tgt1, emb_tgt])
            elif cfg.PGM.USEATTEND == 'attentionshareNoother':
                gmattend_layer = getattr(self, 'gmattend{}'.format(i))
                A_src1 = gmattend_layer(emb_src)
                A_tgt1 = gmattend_layer(emb_tgt)
                emb_src, emb_tgt = gnn_layer([A_src1, emb_src], [A_tgt1, emb_tgt])
            else:
                emb_src, emb_tgt = gnn_layer([A_src, emb_src], [A_tgt, emb_tgt])
            UseAIS = False
            if UseAIS:
                affinity = getattr(self, 'affinity_{}'.format(i))
                s = affinity(emb_src, emb_tgt)
                InstNorm_layer = getattr(self, 'InstNorm_layer_{}'.format(i))
                s = InstNorm_layer(s[:,None,:,:]).squeeze(dim=1)
            else:
                featmatch = match_features(emb_src, emb_tgt)
                s = compute_affinity(featmatch)
            log_s = sinkhorn_rpm(s, n_iters=20, slack=cfg.PGM.SKADDCR)
            s = torch.exp(log_s)

            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1_new = cross_graph(torch.cat((emb_src, torch.bmm(s, emb_tgt)), dim=-1))
                emb2_new = cross_graph(torch.cat((emb_tgt, torch.bmm(s.transpose(1, 2), emb_src)), dim=-1))
                emb_src = emb1_new
                emb_tgt = emb2_new

        # if self.classify_layer is None:
        #     srcinlier_s = None
        #     refinlier_s = None
        # else:
        #     # srcinlier_s = self.classify_layer(Edge_src, Edge_tgt)
        #     # refinlier_s = self.classify_layer(Edge_tgt, Edge_src)
        #     # srcinlier_s = self.classify_layer(emb_src)
        #     # refinlier_s = self.classify_layer(emb_tgt)
        #     # srcinlier_s = None
        #     # refinlier_s = None
        if cfg.DATASET.NOISE_TYPE != 'clean':
            srcinlier_s = torch.sum(s, dim=-1, keepdim=True)
            refinlier_s = torch.sum(s, dim=-2)[:, :, None]
        else:
            srcinlier_s = None
            refinlier_s = None

        if torch.sum(torch.isnan(s)):
            print('discover nan value')
        return s, srcinlier_s, refinlier_s
