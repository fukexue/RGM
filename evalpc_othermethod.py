import torch
import time
from datetime import datetime
from pathlib import Path
import numpy as np

from data.data_loader import get_dataloader, get_datasets
from utils.config import cfg
from utils.evaluation_metric import summarize_metrics, print_metrics, compute_othermethod_metrics
from collections import defaultdict


def eval_model(othermethodtransform, dataloader, metric_is_save=False,
               viz=None, usepgm=True, userefine=False, save_filetime='time'):
    print('-----------------Start evaluation-----------------')
    since = time.time()
    all_val_metrics_np = defaultdict(list)
    iter_num = 0
    num_processed = 0

    dataset_size = len(dataloader.dataset)
    print('train datasize: {}'.format(dataset_size))

    running_since = time.time()

    for inputs in dataloader:
        P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]
        T1_gt, T2_gt = [_.cuda() for _ in inputs['Ts']]
        Label = torch.tensor([_ for _ in inputs['label']])
        P1_gt_copy = P1_gt.clone()
        P2_gt_copy = P2_gt.clone()

        batch_cur_size = P1_gt.size(0)
        iter_num = iter_num + 1

        cur_transform = torch.from_numpy(othermethodtransform[num_processed:num_processed+batch_cur_size]).cuda()
        perform_metrics = compute_othermethod_metrics(cur_transform, P1_gt_copy[:, :, :3], P2_gt_copy[:, :, :3],
                                                      T1_gt[:, :3, :3],T1_gt[:, :3, 3],viz=viz, usepgm=usepgm,
                                                      userefine=userefine)
        num_processed += batch_cur_size

        for k in perform_metrics:
            all_val_metrics_np[k].append(perform_metrics[k])
        all_val_metrics_np['label'].append(Label)

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


if __name__ == '__main__':
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_argspc import parse_args
    from utils.print_easydict import print_easydict
    from utils.visdomshow import VisdomViz
    import socket

    args = parse_args('Point could registration of graph matching evaluation code.')

    if cfg.VISDOM.OPEN:
        hostname = socket.gethostname()
        Visdomins = VisdomViz(env_name=hostname+'Other_Eval', server=cfg.VISDOM.SERVER, port=cfg.VISDOM.PORT)
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
    othermethodtransform = np.load(str(Path(cfg.OUTPUT_PATH) / (cfg.EXPERIMENT.OTHERMETHODFILE)))
    if len(othermethodtransform.shape)>3:
        othermethodtransform = othermethodtransform[:,4]
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        metrics = eval_model(othermethodtransform, dataloader,
                             metric_is_save=True,
                             viz=Visdomins,
                             usepgm=cfg.EXPERIMENT.USEPGM, userefine=cfg.EXPERIMENT.USEREFINE,
                             save_filetime=now_time)
