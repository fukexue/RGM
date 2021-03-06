import numpy as np
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import os

# 画补充材料中的预测的对应点对之间距离的直方图

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
shapenamepath = os.path.join(ROOT_DIR,'data3d/modelnet40_2048/shape_names.txt')
shapename = np.loadtxt(shapenamepath, dtype=str)
pgm_clean = np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_clean/eval_log_2020-11-19-13-42-57_metric.npy'),allow_pickle=True).item()
pgm_jitter = np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_jitter/eval_log_2020-11-19-13-54-41_metric.npy'),allow_pickle=True).item()
pgm_crop = np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/eval_log_2020-11-19-19-54-28_metric.npy'),allow_pickle=True).item()
pgm_unseen_crop = np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Unseen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/eval_log_2020-11-19-20-04-16_metric.npy'),allow_pickle=True).item()
alldata = [pgm_clean, pgm_jitter, pgm_crop, pgm_unseen_crop]
pgmname = ['clean', 'jitter', 'crop', 'unseen_crop']
# alldata = [pgm_unseen_crop]
# pgmname = ['unseen_crop']
for j in range(len(alldata)):
    pre_gt = defaultdict(list)
    pre_dis_gt = defaultdict(list)
    pgm_result_data = alldata[j]
    for i in range(len(pgm_result_data['label'])):
        pre_gt[str(pgm_result_data['label'][i])].append(pgm_result_data['cpd_dis_nomean'][i])
        pre_dis_gt[str(pgm_result_data['label'][i])]=np.concatenate([pre_dis_gt[str(pgm_result_data['label'][i])],
                                                                     pgm_result_data['cpd_dis_nomean'][i][~np.isnan(pgm_result_data['cpd_dis_nomean'][i])]])
    for cate_i in range(40):
        if not os.path.exists('fig/supply_cate/'+pgmname[j]):
            os.mkdir('fig/supply_cate/'+pgmname[j])
        plt.hist(pre_dis_gt[str(cate_i)],bins=20)
        myfontdict = {'fontsize': 17, 'color': 'r', 'weight': 'bold'}
        # plt.title(shapename[cate_i], fontdict=myfontdict)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('corresponding error', fontsize=15)
        plt.ylabel('number of correspondences', fontsize=15)
        plt.show()
        plt.savefig('fig/supply_cate/'+pgmname[j]+'/cate_' + str(cate_i) + '_corresp_dis.pdf', format='pdf', bbox_inches='tight', transparent=True,
                    dpi=1200)
        plt.close()
        # fileHandle = open('fig/supply_cate/'+pgmname[j]+'/cate_' + str(cate_i) + '_corresp_dis.svg')
        # svg = fileHandle.read()
        # fileHandle.close()
        # exportFileHandle = open('fig/supply_cate/'+pgmname[j]+'/cate_' + str(cate_i) + '_corresp_dis.pdf', 'w')
        # cairosvg.svg2pdf(bytestring=svg, write_to=exportFileHandle)
        # exportFileHandle.close()

print('end')
