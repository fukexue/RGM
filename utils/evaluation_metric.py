import torch
import open3d
import numpy as np
from typing import Dict, List
from utils import dcputil
from utils.se3 import transform
from utils.config import cfg
from models.correspondSlover import SVDslover
# from probreg import filterreg, cpd, gmmtree
# import copy


def to_numpy(tensor):
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise NotImplementedError

def pck(x, x_gt, perm_mat, dist_threshs, ns):
    """
    Percentage of Correct Keypoints evaluation metric.
    :param x: candidate coordinates
    :param x_gt: ground truth coordinates
    :param perm_mat: permutation matrix or doubly stochastic matrix indicating correspondence
    :param dist_threshs: a iterable list of thresholds in pixel
    :param ns: number of exact pairs.
    :return: pck, matched num of pairs, total num of pairs
    """
    device = x.device
    batch_num = x.shape[0]
    thresh_num = dist_threshs.shape[1]

    indices = torch.argmax(perm_mat, dim=-1)

    dist = torch.zeros(batch_num, x_gt.shape[1], device=device)
    for b in range(batch_num):
        x_correspond = x[b, indices[b], :]
        dist[b, 0:ns[b]] = torch.norm(x_correspond - x_gt[b], p=2, dim=-1)[0:ns[b]]

    match_num = torch.zeros(thresh_num, device=device)
    total_num = torch.zeros(thresh_num, device=device)
    for b in range(batch_num):
        for idx in range(thresh_num):
            matches = (dist[b] < dist_threshs[b, idx])[0:ns[b]]
            match_num[idx] += torch.sum(matches).to(match_num.dtype)
            total_num[idx] += ns[b].to(total_num.dtype)

    return match_num / total_num, match_num, total_num


def matching_accuracy(pmat_pred, pmat_gt, ns):
    """
    Matching Accuracy between predicted permutation matrix and ground truth permutation matrix.
    :param pmat_pred: predicted permutation matrix
    :param pmat_gt: ground truth permutation matrix
    :param ns: number of exact pairs
    :return: matching accuracy, matched num of pairs, total num of pairs
    """
    device = pmat_pred.device
    batch_num = pmat_pred.shape[0]

    pmat_gt = pmat_gt.to(device)

    assert torch.all((pmat_pred == 0) + (pmat_pred == 1)), 'pmat_pred can noly contain 0/1 elements.'
    assert torch.all((pmat_gt == 0) + (pmat_gt == 1)), 'pmat_gt should noly contain 0/1 elements.'
    assert torch.all(torch.sum(pmat_pred, dim=-1) <= 1) and torch.all(torch.sum(pmat_pred, dim=-2) <= 1)
    assert torch.all(torch.sum(pmat_gt, dim=-1) <= 1) and torch.all(torch.sum(pmat_gt, dim=-2) <= 1)

    #indices_pred = torch.argmax(pmat_pred, dim=-1)
    #indices_gt = torch.argmax(pmat_gt, dim=-1)

    #matched = (indices_gt == indices_pred).type(pmat_pred.dtype)
    match_num_list = []
    gt_num_list = []
    pred_num_list = []
    acc_gt = []
    acc_pred = []
    for b in range(batch_num):
        #match_num += torch.sum(matched[b, :ns[b]])
        #total_num += ns[b].item()
        match_num = torch.sum(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]]) + 1e-8
        gt_num = torch.sum(pmat_gt[b, :ns[b]]) + 1e-8
        pred_num = torch.sum(pmat_pred[b, :ns[b]]) + 1e-8
        match_num_list.append(match_num.cpu().numpy())
        gt_num_list.append(gt_num.cpu().numpy())
        pred_num_list.append(pred_num.cpu().numpy())
        acc_gt.append((match_num/gt_num).cpu().numpy())
        acc_pred.append((match_num/pred_num).cpu().numpy())

    return {'acc_gt':    np.array(acc_gt),
            'acc_pred':  np.array(acc_pred),
            'match_num': np.array(match_num_list),
            'gt_num':    np.array(gt_num_list),
            'pred_num':  np.array(pred_num_list)}


def calcorrespondpc(pmat_pred, pc2_gt):
    pc2 = torch.zeros_like(pc2_gt).to(pc2_gt)
    pmat_pred_index = np.zeros((pc2_gt.shape[0], pc2_gt.shape[1]), dtype=int)
    for i in range(pmat_pred.shape[0]):
        pmat_predi_index1 = torch.where(pmat_pred[i])
        pmat_predi_index00 = torch.where(torch.sum(pmat_pred[i], dim=0) == 0)[0]  #n row sum->1，1024
        pmat_predi_index01 = torch.where(torch.sum(pmat_pred[i], dim=1) == 0)[0]  #n col sum->1024,1
        pc2[i, torch.cat((pmat_predi_index1[0], pmat_predi_index01))] = \
            pc2_gt[i, torch.cat((pmat_predi_index1[1], pmat_predi_index00))]
        # pmat_pred_index[i] = torch.cat((pmat_predi_index1[1], pmat_predi_index00)).cpu().numpy()
        pmat_pred_index[i, pmat_predi_index1[0].cpu().numpy()] = pmat_predi_index1[1].cpu().numpy()
        pmat_pred_index[i, pmat_predi_index01.cpu().numpy()] = pmat_predi_index00.cpu().numpy()
    return pc2, pmat_pred_index


def square_distance(src, dst):
        return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)


def refine_reg(initR, initt, src_gt, tgt_gt):
    src_svd = transform(torch.cat((initR, initt[:, :, None]), dim=2).detach().cpu().numpy(),
                       src_gt.detach().cpu().numpy())
    tgt_gt_np = tgt_gt.cpu().numpy()
    re_R = torch.zeros_like(initR)
    re_t = torch.zeros_like(initt)
    for src, tgt, index in zip(src_svd, tgt_gt_np, range(tgt_gt_np.shape[0])):
        src_o3 = open3d.geometry.PointCloud()
        tgt_o3 = open3d.geometry.PointCloud()
        src_o3.points = open3d.utility.Vector3dVector(src)
        tgt_o3.points = open3d.utility.Vector3dVector(tgt)
        filterregresult = open3d.registration.registration_icp(src_o3, tgt_o3, cfg.EXPERIMENT.ICPMAXCDIST)
        re_R[index] = torch.from_numpy(filterregresult.transformation[:3,:3])
        re_t[index] = torch.from_numpy(filterregresult.transformation[:3,3])
    final_R = torch.bmm(re_R, initR)
    final_t = (torch.bmm(re_R, initt[:, :, None]) + re_t[:, :, None]).squeeze(-1)
    return final_R, final_t


def compute_transform(s_perm_mat, P1_gt, P2_gt, R_gt, T_gt, viz=None, usepgm=True, userefine=False):
    if usepgm:
        pre_P2_gt, _ = calcorrespondpc(s_perm_mat, P2_gt)
        R_pre, T_pre = SVDslover(P1_gt.clone(), pre_P2_gt, s_perm_mat)
    else:
        pre_P2_gt = P2_gt
        R_pre = torch.eye(3,3).repeat(P1_gt.shape[0], 1, 1).to(P2_gt)
        T_pre = torch.zeros_like(T_gt).to(P2_gt)
    if userefine:
        R_pre, T_pre = refine_reg(R_pre, T_pre, P1_gt, pre_P2_gt)
    return R_pre, T_pre


def compute_metrics(s_perm_mat, P1_gt, P2_gt, R_gt, T_gt, viz=None, usepgm=True, userefine=False):
    # compute r,t
    R_pre, T_pre = compute_transform(s_perm_mat, P1_gt, P2_gt, R_gt, T_gt, viz=viz, usepgm=usepgm, userefine=userefine)

    r_pre_euler_deg = dcputil.npmat2euler(R_pre.detach().cpu().numpy(), seq='xyz')
    r_gt_euler_deg = dcputil.npmat2euler(R_gt.detach().cpu().numpy(), seq='xyz')
    r_mse = np.mean((r_gt_euler_deg - r_pre_euler_deg) ** 2, axis=1)
    r_mae = np.mean(np.abs(r_gt_euler_deg - r_pre_euler_deg), axis=1)
    t_mse = torch.mean((T_gt - T_pre) ** 2, dim=1)
    t_mae = torch.mean(torch.abs(T_gt - T_pre), dim=1)

    # Rotation, translation errors (isotropic, i.e. doesn't depend on error
    # direction, which is more representative of the actual error)
    concatenated = dcputil.concatenate(dcputil.inverse(R_gt.cpu().numpy(), T_gt.cpu().numpy()),
                                       np.concatenate([R_pre.cpu().numpy(), T_pre.unsqueeze(-1).cpu().numpy()],
                                                      axis=-1))
    rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
    residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
    residual_transmag = concatenated[:, :, 3].norm(dim=-1)

    # Chamfer distance
    # src_transformed = transform(pred_transforms, points_src)
    P1_transformed = torch.from_numpy(transform(torch.cat((R_pre, T_pre[:, :, None]), dim=2).detach().cpu().numpy(),
                                                P1_gt.detach().cpu().numpy())).to(P1_gt)
    dist_src = torch.min(square_distance(P1_transformed, P2_gt), dim=-1)[0]
    dist_ref = torch.min(square_distance(P2_gt, P1_transformed), dim=-1)[0]
    chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

    # Source distance
    P1_pre_trans = torch.from_numpy(transform(torch.cat((R_pre, T_pre[:, :, None]), dim=2).detach().cpu().numpy(),
                                              P1_gt.detach().cpu().numpy())).to(P1_gt)
    P1_gt_trans = torch.from_numpy(transform(torch.cat((R_gt, T_gt[:, :, None]), dim=2).detach().cpu().numpy(),
                                             P1_gt.detach().cpu().numpy())).to(P1_gt)
    dist_src = torch.min(square_distance(P1_pre_trans, P1_gt_trans), dim=-1)[0]
    presrc_dist = torch.mean(dist_src, dim=1)

    # Clip Chamfer distance
    clip_val = torch.Tensor([0.1]).cuda()
    P1_transformed = torch.from_numpy(transform(torch.cat((R_pre, T_pre[:, :, None]), dim=2).detach().cpu().numpy(),
                                                P1_gt.detach().cpu().numpy())).to(P1_gt)
    dist_src = torch.min(torch.min(torch.sqrt(square_distance(P1_transformed, P2_gt)), dim=-1)[0], clip_val)
    dist_ref = torch.min(torch.min(torch.sqrt(square_distance(P2_gt, P1_transformed)), dim=-1)[0], clip_val)
    clip_chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

    # correspondence distance
    P2_gt_copy, _ = calcorrespondpc(s_perm_mat, P2_gt.detach())
    inlier_src = torch.sum(s_perm_mat, axis=-1)[:, :, None]
    # inlier_ref = torch.sum(s_perm_mat, axis=-2)[:, :, None]
    P1_gt_trans_corr = P1_gt_trans.mul(inlier_src)
    P2_gt_copy_coor = P2_gt_copy.mul(inlier_src)
    correspond_dis=torch.sqrt(torch.sum((P1_gt_trans_corr-P2_gt_copy_coor)**2, dim=-1, keepdim=True))
    correspond_dis[inlier_src == 0] = np.nan

    metrics = {'r_mse': r_mse,
               'r_mae': r_mae,
               't_mse': to_numpy(t_mse),
               't_mae': to_numpy(t_mae),
               'err_r_deg': to_numpy(residual_rotdeg),
               'err_t': to_numpy(residual_transmag),
               'chamfer_dist': to_numpy(chamfer_dist),
               'pcab_dist': to_numpy(presrc_dist),
               'clip_chamfer_dist': to_numpy(clip_chamfer_dist),
               'pre_transform':np.concatenate((to_numpy(R_pre),to_numpy(T_pre)[:,:,None]),axis=2),
               'gt_transform':np.concatenate((to_numpy(R_gt),to_numpy(T_gt)[:,:,None]),axis=2),
               'cpd_dis_nomean':to_numpy(correspond_dis)}

    return metrics


def compute_othermethod_metrics(transform_para, P1_gt, P2_gt, R_gt, T_gt, viz=None, usepgm=True, userefine=False):
    # compute r,t
    R_pre, T_pre = transform_para[:,:3,:3], transform_para[:,:3,3]

    r_pre_euler_deg = dcputil.npmat2euler(R_pre.detach().cpu().numpy(), seq='xyz')
    r_gt_euler_deg = dcputil.npmat2euler(R_gt.detach().cpu().numpy(), seq='xyz')
    r_mse = np.mean((r_gt_euler_deg - r_pre_euler_deg) ** 2, axis=1)
    r_mae = np.mean(np.abs(r_gt_euler_deg - r_pre_euler_deg), axis=1)
    t_mse = torch.mean((T_gt - T_pre) ** 2, dim=1)
    t_mae = torch.mean(torch.abs(T_gt - T_pre), dim=1)

    # Rotation, translation errors (isotropic, i.e. doesn't depend on error
    # direction, which is more representative of the actual error)
    concatenated = dcputil.concatenate(dcputil.inverse(R_gt.cpu().numpy(), T_gt.cpu().numpy()),
                                       np.concatenate([R_pre.cpu().numpy(), T_pre.unsqueeze(-1).cpu().numpy()],
                                                      axis=-1))
    rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
    residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
    residual_transmag = concatenated[:, :, 3].norm(dim=-1)

    # Chamfer distance
    # src_transformed = transform(pred_transforms, points_src)
    P1_transformed = torch.from_numpy(transform(torch.cat((R_pre, T_pre[:, :, None]), dim=2).detach().cpu().numpy(),
                                                P1_gt.detach().cpu().numpy())).to(P1_gt)
    dist_src = torch.min(square_distance(P1_transformed, P2_gt), dim=-1)[0]
    dist_ref = torch.min(square_distance(P2_gt, P1_transformed), dim=-1)[0]
    chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

    P1_pre_trans = torch.from_numpy(transform(torch.cat((R_pre, T_pre[:, :, None]), dim=2).detach().cpu().numpy(),
                                              P1_gt.detach().cpu().numpy())).to(P1_gt)
    P1_gt_trans = torch.from_numpy(transform(torch.cat((R_gt, T_gt[:, :, None]), dim=2).detach().cpu().numpy(),
                                             P1_gt.detach().cpu().numpy())).to(P1_gt)
    dist_src = torch.min(square_distance(P1_pre_trans, P1_gt_trans), dim=-1)[0]
    presrc_dist = torch.mean(dist_src, dim=1)

    # Clip Chamfer distance
    clip_val = torch.Tensor([0.1]).cuda()
    P1_transformed = torch.from_numpy(transform(torch.cat((R_pre, T_pre[:, :, None]), dim=2).detach().cpu().numpy(),
                                                P1_gt.detach().cpu().numpy())).to(P1_gt)
    dist_src = torch.min(torch.min(torch.sqrt(square_distance(P1_transformed, P2_gt)), dim=-1)[0], clip_val)
    dist_ref = torch.min(torch.min(torch.sqrt(square_distance(P2_gt, P1_transformed)), dim=-1)[0], clip_val)
    clip_chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

    metrics = {'r_mse': r_mse,
               'r_mae': r_mae,
               't_mse': to_numpy(t_mse),
               't_mae': to_numpy(t_mae),
               'err_r_deg': to_numpy(residual_rotdeg),
               'err_t': to_numpy(residual_transmag),
               'chamfer_dist': to_numpy(chamfer_dist),
               'pcab_dist': to_numpy(presrc_dist),
               'clip_chamfer_dist': to_numpy(clip_chamfer_dist),
               'pre_transform': np.concatenate((to_numpy(R_pre), to_numpy(T_pre)[:, :, None]), axis=2),
               'gt_transform': np.concatenate((to_numpy(R_gt), to_numpy(T_gt)[:, :, None]), axis=2)}

    return metrics


def summarize_metrics(metrics):
    """Summaries computed metrices by taking mean over all data instances"""
    summarized = {}
    for k in metrics:
        if k.endswith('mse'):
            summarized[k[:-3] + 'rmse'] = np.sqrt(np.mean(metrics[k]))
        elif k.startswith('err'):
            summarized[k + '_mean'] = np.mean(metrics[k])
            summarized[k + '_rmse'] = np.sqrt(np.mean(metrics[k]**2))
        elif k.endswith('nomean'):
            summarized[k] = metrics[k]
        else:
            summarized[k] = np.mean(metrics[k])

    return summarized


def print_metrics(summary_metrics: Dict, title: str = 'Metrics'):
    """Prints out formated metrics to logger"""

    print('=' * (len(title) + 1))
    print(title + ':')

    print('DeepCP metrics:{:.4f}(rot-rmse) | {:.4f}(rot-mae) | {:.4g}(trans-rmse) | {:.4g}(trans-mae)'.
          format(summary_metrics['r_rmse'], summary_metrics['r_mae'],
                 summary_metrics['t_rmse'], summary_metrics['t_mae'],
    ))
    print('Rotation error {:.4f}(deg, mean) | {:.4f}(deg, rmse)'.
          format(summary_metrics['err_r_deg_mean'],
                 summary_metrics['err_r_deg_rmse']))
    print('Translation error {:.4g}(mean) | {:.4g}(rmse)'.
          format(summary_metrics['err_t_mean'],
                 summary_metrics['err_t_rmse']))
    print('RPM Chamfer error: {:.7f}(mean-sq)'.
          format(summary_metrics['chamfer_dist']))
    print('Source error: {:.7f}(mean-sq)'.
          format(summary_metrics['pcab_dist']))
    print('Clip Chamfer error: {:.7f}(mean-sq)'.
          format(summary_metrics['clip_chamfer_dist']))
