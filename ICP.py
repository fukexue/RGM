import torch
import time
from datetime import datetime
from pathlib import Path
import numpy as np

from data.data_loader import get_dataloader, get_datasets
from utils.config import cfg
from utils.evaluation_metric import summarize_metrics, print_metrics, to_numpy, square_distance, compute_transform
from collections import defaultdict
from utils import dcputil
from utils.se3 import transform


def eval_model(dataloader, metric_is_save=False, viz=None, usepgm=True,
               userefine=False, save_filetime='time'):
    print('-----------------Start evaluation-----------------')
    since = time.time()
    all_val_metrics_np = defaultdict(list)
    iter_num = 0

    dataset_size = len(dataloader.dataset)
    print('train datasize: {}'.format(dataset_size))

    running_since = time.time()

    for inputs in dataloader:
        P1_gt, P2_gt = [_ for _ in inputs['Ps']]
        perm_mat = inputs['gt_perm_mat']
        T1_gt, T2_gt = [_ for _ in inputs['Ts']]
        Label = torch.tensor([_ for _ in inputs['label']])

        batch_cur_size = perm_mat.size(0)
        iter_num = iter_num + 1
        infer_time = time.time()

        perform_metrics = compute_metrics_icp(None, P1_gt[:,:,:3], P2_gt[:,:,:3], T1_gt[:,:3,:3], T1_gt[:,:3,3],
                                              viz=viz, usepgm=usepgm, userefine=userefine)
        infer_time = time.time() - infer_time

        for k in perform_metrics:
            all_val_metrics_np[k].append(perform_metrics[k])
        all_val_metrics_np['label'].append(Label)
        all_val_metrics_np['infertime'].append(np.repeat(infer_time / batch_cur_size, batch_cur_size))

        if iter_num % cfg.STATISTIC_STEP == 0 and metric_is_save:
            running_speed = cfg.STATISTIC_STEP * batch_cur_size / (time.time() - running_since)
            print('Iteration {:<4} {:>4.2f}sample/s'.format(iter_num, running_speed))
            running_since = time.time()

    all_val_metrics_np = {k: np.concatenate(all_val_metrics_np[k]) for k in all_val_metrics_np}
    summary_metrics = summarize_metrics(all_val_metrics_np)
    print_metrics(summary_metrics)
    if metric_is_save:
        np.save(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + save_filetime + '_metric')),
                all_val_metrics_np)

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return summary_metrics


def compute_metrics_icp(s_perm_mat, P1_gt, P2_gt, R_gt, T_gt, viz=None, usepgm=True, userefine=False):
    # compute r,t
    R_pre, T_pre = compute_transform(s_perm_mat, P1_gt, P2_gt, R_gt, T_gt, viz=viz, usepgm=usepgm, userefine=userefine)

    r_pre_euler_deg = dcputil.npmat2euler(R_pre.detach().cpu().numpy(), seq='xyz')
    r_gt_euler_deg = dcputil.npmat2euler(R_gt.detach().cpu().numpy(), seq='xyz')
    r_mse = np.mean((r_gt_euler_deg - r_pre_euler_deg) ** 2, axis=1)
    r_mae = np.mean(np.abs(r_gt_euler_deg - r_pre_euler_deg), axis=1)
    t_mse = torch.mean((T_gt - T_pre) ** 2, dim=1)
    t_mae = torch.mean(torch.abs(T_gt - T_pre), dim=1)

    # if viz is not None:
    #     P1_pre = transform(torch.cat((R_pre, T_pre[:,:, None]), dim=2).detach().cpu().numpy(),
    #                        P1_gt.cpu().numpy())
    #     P1_gt_pre = transform(torch.cat((R_gt, T_gt[:,:, None]), dim=2).detach().cpu().numpy(),
    #                        P1_gt.cpu().numpy())
    #     for pi in range(P1_pre.shape[0]):
    #         if r_mae[pi]>20:
    #             inlier_src = np.sum(s_perm_mat.cpu().numpy(), axis=-1)[:, :, None]
    #             inlier_ref = np.sum(s_perm_mat.cpu().numpy(), axis=-2)[:, :, None]
    #             outlier_src = 1 - np.sum(s_perm_mat.cpu().numpy(), axis=-1)[:, :, None]
    #             outlier_ref = 1 - np.sum(s_perm_mat.cpu().numpy(), axis=-2)[:, :, None]
    #             src_corr = P1_pre[pi] * inlier_src[pi]
    #             srcgt_corr = P1_gt_pre[pi] * inlier_src[pi]
    #             tgt_coor = P2_gt[pi].cpu().numpy() * inlier_ref[pi]
    #             src_nocoor = P1_pre[pi] * outlier_src[pi]
    #             srcgt_nocoor = P1_gt_pre[pi] * outlier_src[pi]
    #             tgt_nocoor = P2_gt[pi].cpu().numpy() * outlier_ref[pi]
    #             viz.pc2_show(torch.from_numpy(src_corr), torch.from_numpy(tgt_coor), title='1')
    #             viz.pc2_show(torch.from_numpy(srcgt_corr), torch.from_numpy(tgt_coor), title='2')
    #             viz.pc4_show(torch.from_numpy(src_corr), torch.from_numpy(tgt_coor), torch.from_numpy(src_nocoor),
    #                          torch.from_numpy(tgt_nocoor), title='3')
    #             viz.pc4_show(torch.from_numpy(srcgt_corr), torch.from_numpy(tgt_coor), torch.from_numpy(srcgt_nocoor),
    #                          torch.from_numpy(tgt_nocoor), title='4')
    #
    #             # viz.pc2_show(torch.from_numpy(P1_pre[pi]), P2_gt[pi].cpu(), title='p1_g2')
    #             # viz.pc2_show(torch.from_numpy(P1_gt_pre[pi]), P2_gt[pi].cpu(), title='p1g_g2')
    #             # viz.pc2_show(torch.from_numpy(P1_gt_pre[pi]), torch.from_numpy(P1_pre[pi]), title='p1g_p1')

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
    clip_val = torch.Tensor([0.1])
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


if __name__ == '__main__':
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_argspc import parse_args
    from utils.print_easydict import print_easydict
    from utils.visdomshow import VisdomViz
    import socket

    args = parse_args('Point could registration of graph matching evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    if cfg.VISDOM.OPEN:
        hostname = socket.gethostname()
        Visdomins = VisdomViz(env_name=hostname+'PGM_Eval', server=cfg.VISDOM.SERVER, port=cfg.VISDOM.PORT)
        Visdomins.viz.close()
    else:
        Visdomins = None

    torch.manual_seed(cfg.RANDOM_SEED)

    pc_dataset = get_datasets(partition = 'test',
                              num_points = cfg.DATASET.POINT_NUM,
                              unseen = cfg.DATASET.UNSEEN,
                              noise_type = cfg.DATASET.NOISE_TYPE,
                              rot_mag = cfg.DATASET.ROT_MAG,
                              trans_mag = cfg.DATASET.TRANS_MAG,
                              partial_p_keep = cfg.DATASET.PARTIAL_P_KEEP)

    dataloader = get_dataloader(pc_dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        metrics = eval_model(dataloader,
                             metric_is_save=True,
                             viz=Visdomins,
                             usepgm=cfg.EXPERIMENT.USEPGM, userefine=cfg.EXPERIMENT.USEREFINE,
                             save_filetime=now_time)
