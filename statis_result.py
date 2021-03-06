import os
import numpy as np
from typing import Dict
import pandas as pd
from utils.dup_stdout_manager import DupStdoutFileManager
from utils.evaluation_metric import summarize_metrics


def print_metrics(summary_metrics: Dict, title: str = 'Metrics'):
    """Prints out formated metrics to logger"""

    print('=' * (len(title) + 1))
    print(title + ':')

    # if losses_by_iteration is not None:
    #     losses_all_str = ' | '.join(['{:.5f}'.format(c) for c in losses_by_iteration])
    #     logger.info('Losses by iteration: {}'.format(losses_all_str))

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
          format(summary_metrics['clip_chamfer_dist']) if 'clip_chamfer_dist' in summary_metrics else 'Clip Chamfer error:')


def output2dict(filename, summary_metrics: Dict, xlsx_data):
    xlsx_data['file_name'].append(filename)
    xlsx_data['r_rmse'].append(summary_metrics['r_rmse'])
    xlsx_data['t_rmse'].append(summary_metrics['t_rmse'])
    xlsx_data['err_r_deg_rmse'].append(summary_metrics['err_r_deg_rmse'])
    xlsx_data['err_t_rmse'].append(summary_metrics['err_t_rmse'])
    xlsx_data['chamfer_dist'].append(summary_metrics['chamfer_dist'])
    xlsx_data['source_dist'].append(summary_metrics['pcab_dist'])
    xlsx_data['r_mae'].append(summary_metrics['r_mae'])
    xlsx_data['t_mae'].append(summary_metrics['t_mae'])
    xlsx_data['err_r_deg_mean'].append(summary_metrics['err_r_deg_mean'])
    xlsx_data['err_t_mean'].append(summary_metrics['err_t_mean'])
    xlsx_data['CCD'].append(summary_metrics['clip_chamfer_dist'] if 'clip_chamfer_dist' in summary_metrics else 0)
    return xlsx_data


def xlsx_init(xlsx_data):
    xlsx_data['file_name'] = []
    xlsx_data['r_rmse'] = []
    xlsx_data['t_rmse'] = []
    xlsx_data['err_r_deg_rmse'] = []
    xlsx_data['err_t_rmse'] = []
    xlsx_data['chamfer_dist'] = []
    xlsx_data['source_dist'] = []
    xlsx_data['r_mae'] = []
    xlsx_data['t_mae'] = []
    xlsx_data['err_r_deg_mean'] = []
    xlsx_data['err_t_mean'] = []
    xlsx_data['CCD'] = []
    xlsx_data['acc'] = []
    xlsx_data['acc'] = []
    return xlsx_data


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(ROOT_DIR,'output')
allalgfile = np.sort(os.listdir(output_dir))
xlsx_data = xlsx_init({})

for file in allalgfile:
    with DupStdoutFileManager(output_dir+'/'+ file + '/acc_record.log') as _:
        npyfile = np.sort([txt for txt in os.listdir(output_dir+'/'+file) if txt.endswith('metric.npy')])
        for npyfile_i in npyfile:
            metric_all = np.load(output_dir + '/' + file + '/' + npyfile_i, allow_pickle=True).item()
            print(file+'/'+npyfile_i)
            summary_metrics = summarize_metrics(metric_all)
            print_metrics(summary_metrics)
            xlsx_data = output2dict(file+'/'+npyfile_i, summary_metrics, xlsx_data)
            for i in [0.1,0.5,1,5,10,20]:
                cur_acc = sum((metric_all['r_mae'] <= i)*(metric_all['t_mae'] <= 0.1)) / metric_all['r_mae'].size
                print('a[\'r_mae\']<={}, num:{} acc:{:.5f}'.format(i, sum(metric_all['r_mae'] <= i), cur_acc))
                if i==1:
                    xlsx_data['acc'].append(cur_acc)
            print('')
df = pd.DataFrame(xlsx_data)
df.to_excel(os.path.join(ROOT_DIR,'xls_result/result.xlsx'))
print('end')