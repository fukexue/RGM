import torch
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import scipy.io as io

from data.data_loader import get_dataloader, get_datasets
from utils.config import cfg
from collections import defaultdict


def to_numpy(tensor):
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise NotImplementedError


def eval_model(dataloader, creat_data_path, metric_is_save=False, save_filetime='time'):
    print('-----------------Creat dataset-----------------')
    since = time.time()
    all_val_metrics_np = defaultdict(list)
    iter_num = 0

    dataset_size = len(dataloader.dataset)
    print('train datasize: {}'.format(dataset_size))

    running_since = time.time()

    for inputs in dataloader:
        P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]
        perm_mat = inputs['gt_perm_mat'].cuda()
        T1_gt, T2_gt = [_.cuda() for _ in inputs['Ts']]
        Label = torch.tensor([_ for _ in inputs['label']])

        batch_cur_size = perm_mat.size(0)
        iter_num = iter_num + 1

        all_val_metrics_np['p1'].append(to_numpy(P1_gt))
        all_val_metrics_np['p2'].append(to_numpy(P2_gt))
        all_val_metrics_np['gt_transform'].append(to_numpy(T1_gt))
        all_val_metrics_np['label'].append(Label)
        all_val_metrics_np['raw'].append(to_numpy(inputs['raw'].cuda()))

        if iter_num % cfg.STATISTIC_STEP == 0 and metric_is_save:
            running_speed = cfg.STATISTIC_STEP * batch_cur_size / (time.time() - running_since)
            print('Iteration {:<4} {:>4.2f}sample/s'.format(iter_num, running_speed))
            running_since = time.time()

    all_val_metrics_np = {k: np.concatenate(all_val_metrics_np[k]) for k in all_val_metrics_np}
    if metric_is_save:
        np.save(str(Path(creat_data_path) / ('Creat_dataset_log_' + save_filetime + '_metric')),
                all_val_metrics_np)
        # creat_data_path_split = creat_data_path.split('/')
        io.savemat(creat_data_path+'/Dataset' + save_filetime + '_metric.mat', all_val_metrics_np)

    time_elapsed = time.time() - since
    print('Creat dataset complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    return None


if __name__ == '__main__':
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_argspc import parse_args
    from utils.print_easydict import print_easydict
    from utils.visdomshow import VisdomViz
    import socket

    args = parse_args('Point could registration of graph matching evaluation code.')

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

    op_split = cfg.OUTPUT_PATH.split('/')
    creat_data_path = 'creat_data/'+op_split[1]
    if not Path(creat_data_path).exists():
        Path(creat_data_path).mkdir(parents=True)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with DupStdoutFileManager(str(Path(creat_data_path) / ('Creat_dataset_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        metrics = eval_model(dataloader,creat_data_path,metric_is_save=True,save_filetime=now_time)
