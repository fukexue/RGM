import torch
import time
from datetime import datetime
from pathlib import Path
import numpy as np

from data.data_loader import get_dataloader, get_datasets
from utils.config import cfg
from utils.model_sl import load_model
from utils.hungarian import hungarian
from utils.loss_func import PermLoss
from utils.evaluation_metric import matching_accuracy, calcorrespondpc, summarize_metrics, print_metrics, compute_metrics, compute_transform
from parallel import DataParallel
from models.correspondSlover import SVDslover, RANSACSVDslover
from collections import defaultdict


def eval_model(model, dataloader, eval_epoch=None, metric_is_save=False, estimate_iters=1,
               viz=None, usepgm=True, userefine=False, save_filetime='time'):
    print('-----------------Start evaluation-----------------')
    lap_solver = hungarian
    permevalLoss = PermLoss()
    since = time.time()
    all_val_metrics_np = defaultdict(list)
    iter_num = 0

    dataset_size = len(dataloader.dataset)
    print('train datasize: {}'.format(dataset_size))
    device = next(model.parameters()).device
    print('model on device: {}'.format(device))

    if eval_epoch is not None:
        if eval_epoch == -1:
            model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_best.pt')
            print('Loading best model parameters')
            load_model(model, model_path)
        else:
            model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pt'.format(eval_epoch))
            print('Loading model parameters from {}'.format(model_path))
            load_model(model, model_path)

    was_training = model.training
    model.eval()
    running_since = time.time()

    for inputs in dataloader:
        P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]
        n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]
        A1_gt, A2_gt = [_.cuda() for _ in inputs['As']]
        perm_mat = inputs['gt_perm_mat'].cuda()
        T1_gt, T2_gt = [_.cuda() for _ in inputs['Ts']]
        Inlier_src_gt, Inlier_ref_gt = [_.cuda() for _ in inputs['Ins']]
        Label = torch.tensor([_ for _ in inputs['label']])

        batch_cur_size = perm_mat.size(0)
        iter_num = iter_num + 1
        infer_time = time.time()

        with torch.set_grad_enabled(False):
            if cfg.EVAL.ITERATION:
                P1_gt_copy = P1_gt.clone()
                P2_gt_copy = P2_gt.clone()
                P1_gt_copy_inv = P1_gt.clone()
                P2_gt_copy_inv = P2_gt.clone()
                s_perm_mat = caliters_perm(model, P1_gt_copy, P2_gt_copy, A1_gt, A2_gt, n1_gt, n2_gt, estimate_iters)
                if cfg.EVAL.CYCLE:
                    s_perm_mat_inv = caliters_perm(model, P2_gt_copy_inv, P1_gt_copy_inv, A2_gt, A1_gt, n2_gt, n1_gt, estimate_iters)
                    s_perm_mat = s_perm_mat * s_perm_mat_inv.permute(0, 2, 1)
                permevalloss=torch.tensor([0])
            else:
                s_prem_tensor, Inlier_src_pre, Inlier_ref_pre_tensor = model(P1_gt, P2_gt, A1_gt, A2_gt, n1_gt, n2_gt)
                if cfg.EVAL.CYCLE:
                    s_prem_tensor_inv, Inlier_src_pre_inv, Inlier_ref_pre_tensor_inv = model(P2_gt, P1_gt, A2_gt, A1_gt, n2_gt, n1_gt)

                if cfg.PGM.USEINLIERRATE:
                    s_prem_tensor = Inlier_src_pre * s_prem_tensor * Inlier_ref_pre_tensor.transpose(2, 1).contiguous()
                    if cfg.EVAL.CYCLE:
                        s_prem_tensor_inv = Inlier_src_pre_inv * s_prem_tensor_inv * \
                                            Inlier_ref_pre_tensor_inv.transpose(2,1).contiguous()
                permevalloss = permevalLoss(s_prem_tensor, perm_mat, n1_gt, n2_gt)
                s_perm_mat = lap_solver(s_prem_tensor, n1_gt, n2_gt, Inlier_src_pre, Inlier_ref_pre_tensor)
                if cfg.EVAL.CYCLE:
                    s_perm_mat_inv = lap_solver(s_prem_tensor_inv, n2_gt, n1_gt, Inlier_src_pre_inv, Inlier_ref_pre_tensor_inv)
                    s_perm_mat = s_perm_mat*s_perm_mat_inv.permute(0,2,1)
            #test time
            compute_transform(s_perm_mat, P1_gt[:,:,:3], P2_gt[:,:,:3], T1_gt[:,:3,:3], T1_gt[:,:3,3])

        infer_time = time.time() - infer_time
        match_metrics = matching_accuracy(s_perm_mat, perm_mat, n1_gt)
        perform_metrics = compute_metrics(s_perm_mat, P1_gt[:,:,:3], P2_gt[:,:,:3], T1_gt[:,:3,:3], T1_gt[:,:3,3],
                                          viz=viz, usepgm=usepgm, userefine=userefine)

        for k in match_metrics:
            all_val_metrics_np[k].append(match_metrics[k])
        for k in perform_metrics:
            all_val_metrics_np[k].append(perform_metrics[k])
        all_val_metrics_np['label'].append(Label)
        all_val_metrics_np['loss'].append(np.repeat(permevalloss.item(), batch_cur_size))
        all_val_metrics_np['infertime'].append(np.repeat(infer_time/batch_cur_size, batch_cur_size))

        if iter_num % cfg.STATISTIC_STEP == 0 and metric_is_save:
            running_speed = cfg.STATISTIC_STEP * batch_cur_size / (time.time() - running_since)
            print('Iteration {:<4} {:>4.2f}sample/s'.format(iter_num, running_speed))
            running_since = time.time()

    all_val_metrics_np = {k: np.concatenate(all_val_metrics_np[k]) for k in all_val_metrics_np}
    summary_metrics = summarize_metrics(all_val_metrics_np)
    print('Mean-Loss: {:.4f} GT-Acc:{:.4f} Pred-Acc:{:.4f}'.format(summary_metrics['loss'], summary_metrics['acc_gt'], summary_metrics['acc_pred']))
    print_metrics(summary_metrics)
    if metric_is_save:
        np.save(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + save_filetime + '_metric')),
                all_val_metrics_np)

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.train(mode=was_training)

    return summary_metrics


def caliters_perm(model, P1_gt_copy, P2_gt_copy, A1_gt, A2_gt, n1_gt, n2_gt, estimate_iters):
    lap_solver1 = hungarian
    s_perm_indexs = []
    for estimate_iter in range(estimate_iters):
        s_prem_i, Inlier_src_pre, Inlier_ref_pre = model(P1_gt_copy, P2_gt_copy,
                                                         A1_gt, A2_gt, n1_gt, n2_gt)
        if cfg.PGM.USEINLIERRATE:
            s_prem_i = Inlier_src_pre * s_prem_i * Inlier_ref_pre.transpose(2, 1).contiguous()
        s_perm_i_mat = lap_solver1(s_prem_i, n1_gt, n2_gt, Inlier_src_pre, Inlier_ref_pre)
        P2_gt_copy1, s_perm_i_mat_index = calcorrespondpc(s_perm_i_mat, P2_gt_copy)
        s_perm_indexs.append(s_perm_i_mat_index)
        if cfg.EXPERIMENT.USERANSAC:
            R_pre, T_pre, s_perm_i_mat = RANSACSVDslover(P1_gt_copy[:,:,:3], P2_gt_copy1[:,:,:3], s_perm_i_mat)
        else:
            R_pre, T_pre = SVDslover(P1_gt_copy[:,:,:3], P2_gt_copy1[:,:,:3], s_perm_i_mat)
        P1_gt_copy[:,:,:3] = torch.bmm(P1_gt_copy[:,:,:3], R_pre.transpose(2, 1).contiguous()) + T_pre[:, None, :]
        P1_gt_copy[:,:,3:6] = P1_gt_copy[:,:,3:6] @ R_pre.transpose(-1, -2)
    return s_perm_i_mat


if __name__ == '__main__':
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_argspc import parse_args
    from utils.print_easydict import print_easydict
    from utils.visdomshow import VisdomViz
    import socket

    import scipy
    if np.__version__=='1.19.2' and scipy.__version__=='1.5.0':
        print('It is the same as the paper result')
    else:
        print('May not be the same as the results of the paper')

    args = parse_args('Point could registration of graph matching evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    if cfg.VISDOM.OPEN:
        hostname = socket.gethostname()
        Visdomins = VisdomViz(env_name=hostname+'RGM_Eval', server=cfg.VISDOM.SERVER, port=cfg.VISDOM.PORT)
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

    model = Net()
    model = model.to(device)
    model = DataParallel(model, device_ids=cfg.GPUS)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        metrics = eval_model(model, dataloader,
                             eval_epoch=cfg.EVAL.EPOCH if cfg.EVAL.EPOCH != 0 else None,
                             metric_is_save=True,
                             estimate_iters=cfg.EVAL.ITERATION_NUM,
                             viz=Visdomins,
                             usepgm=cfg.EXPERIMENT.USEPGM, userefine=cfg.EXPERIMENT.USEREFINE,
                             save_filetime=now_time)
