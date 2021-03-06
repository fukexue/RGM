import torch
import torch.nn as nn
import math
from models.dgcnn import DGCNN
from models.gconv import Siamese_Gconv
from models.affinity_layer import Affinity
from models.transformer import Transformer

from utils.config import cfg


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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pointfeaturer = DGCNN(cfg.PGM.FEATURES, cfg.PGM.NEIGHBORSNUM, cfg.PGM.FEATURE_EDGE_CHANNEL)
        self.gnn_layer = cfg.PGM.GNN_LAYER
        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(cfg.PGM.FEATURE_NODE_CHANNEL + cfg.PGM.FEATURE_EDGE_CHANNEL, cfg.PGM.GNN_FEAT)
            else:
                gnn_layer = Siamese_Gconv(cfg.PGM.GNN_FEAT, cfg.PGM.GNN_FEAT)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(cfg.PGM.GNN_FEAT))
            if cfg.PGM.USEATTEND == 'attentiontransformer':
                self.add_module('gmattend{}'.format(i), Transformer(2 * cfg.PGM.FEATURE_EDGE_CHANNEL
                                                                    if i == 0 else cfg.PGM.GNN_FEAT))
            self.add_module('InstNorm_layer_{}'.format(i), nn.InstanceNorm2d(1, affine=True))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), nn.Linear(cfg.PGM.GNN_FEAT * 2, cfg.PGM.GNN_FEAT))
    # @profile
    def forward(self, P_src, P_tgt, A_src, A_tgt, ns_src, ns_tgt):
        # extract feature
        Node_src, Edge_src = self.pointfeaturer(P_src)
        Node_tgt, Edge_tgt = self.pointfeaturer(P_tgt)

        emb_src, emb_tgt = torch.cat((Node_src, Edge_src), dim=1).transpose(1, 2).contiguous(), \
                           torch.cat((Node_tgt, Edge_tgt), dim=1).transpose(1, 2).contiguous()
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            if cfg.PGM.USEATTEND == 'attentiontransformer':
                gmattends_layer = getattr(self, 'gmattend{}'.format(i))
                src_embedding, tgt_embedding = gmattends_layer(emb_src, emb_tgt)
                d_k = src_embedding.size(1)
                scores_src = torch.matmul(src_embedding.transpose(2, 1).contiguous(), src_embedding) / math.sqrt(d_k)
                scores_tgt = torch.matmul(tgt_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
                A_src1 = torch.softmax(scores_src, dim=-1)
                A_tgt1 = torch.softmax(scores_tgt, dim=-1)
                emb_src, emb_tgt = gnn_layer([A_src1, emb_src], [A_tgt1, emb_tgt])
            else:
                emb_src, emb_tgt = gnn_layer([A_src, emb_src], [A_tgt, emb_tgt])
            affinity = getattr(self, 'affinity_{}'.format(i))
            # emb_src_norm = torch.norm(emb_src, p=2, dim=2, keepdim=True).detach()
            # emb_tgt_norm = torch.norm(emb_tgt, p=2, dim=2, keepdim=True).detach()
            # emb_src = emb_src.div(emb_src_norm)
            # emb_tgt = emb_tgt.div(emb_tgt_norm)
            s = affinity(emb_src, emb_tgt)
            InstNorm_layer = getattr(self, 'InstNorm_layer_{}'.format(i))
            s = InstNorm_layer(s[:,None,:,:]).squeeze(dim=1)
            log_s = sinkhorn_rpm(s, n_iters=20, slack=cfg.PGM.SKADDCR)
            s = torch.exp(log_s)

            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1_new = cross_graph(torch.cat((emb_src, torch.bmm(s, emb_tgt)), dim=-1))
                emb2_new = cross_graph(torch.cat((emb_tgt, torch.bmm(s.transpose(1, 2), emb_src)), dim=-1))
                emb_src = emb1_new
                emb_tgt = emb2_new

        if cfg.DATASET.NOISE_TYPE != 'clean':
            srcinlier_s = torch.sum(s, dim=-1, keepdim=True)
            refinlier_s = torch.sum(s, dim=-2)[:, :, None]
        else:
            srcinlier_s = None
            refinlier_s = None

        return s, srcinlier_s, refinlier_s
