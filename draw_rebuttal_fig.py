import os
from utils import draw_figures
import numpy as np
import utils.se3 as se3
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter
import torch

def to_percent(y,position):
    return str(int(100*y))+"%"#这里可以用round（）函数设置取几位小数

#好的图需要手动存
def show_graph(pc, node, size=2, theta_1=0, theta_2=0, c0='#b30000',savename=None, close=True):
    if type(pc) == torch.Tensor:
        pc = pc.detach().cpu().numpy()
    if pc.shape[0]==3:
        pc = pc.transpose()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(pc[:,0], pc[:,1], pc[:,2],s=size, alpha=1, c=c0)
    plt.axis("off")
    ax.view_init(theta_1, theta_2)
    ax.auto_scale_xyz([-1,1],[-1,1],[-1,1])
    for i in range(len(node[0])):
        ax.plot([pc[node[0]][i,0], pc[node[1]][i,0]],
                [pc[node[0]][i,1], pc[node[1]][i,1]],
                [pc[node[0]][i,2], pc[node[1]][i,2]], color='b', alpha=0.4)
    plt.show(fig)
    if savename is not None:
        plt.savefig(savename, format='svg', bbox_inches='tight', transparent=True, dpi=600)
        if close:
            plt.close(fig)


def show_2graph(pc0, pc1, node0, node1, size=1, theta_1=0, theta_2=0, c0='#1f4f89', c1='#b30000', savename=None, close=True):
    if type(pc0) == torch.Tensor:
        pc0 = pc0.detach().cpu().numpy()
    if pc0.shape[0]==3:
        pc0 = pc0.transpose()
    if type(pc1) == torch.Tensor:
        pc1 = pc1.detach().cpu().numpy()
    if pc1.shape[0]==3:
        pc1 = pc1.transpose()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(pc0[:, 0], pc0[:, 1], pc0[:, 2], s=size, c=c0, alpha=1)
    ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], s=size, c=c1, alpha=1)
    for i in range(len(node0[0])):
        ax.plot([pc0[node0[0]][i,0], pc0[node0[1]][i,0]],
                [pc0[node0[0]][i,1], pc0[node0[1]][i,1]],
                [pc0[node0[0]][i,2], pc0[node0[1]][i,2]], color=c0, linewidth = 1)
    for i in range(len(node1[0])):
        ax.plot([pc1[node1[0]][i,0], pc1[node1[1]][i,0]],
                [pc1[node1[0]][i,1], pc1[node1[1]][i,1]],
                [pc1[node1[0]][i,2], pc1[node1[1]][i,2]], color=c1, linewidth = 1)
    plt.axis("off")
    ax.view_init(theta_1, theta_2)
    ax.auto_scale_xyz([-1,1],[-1,1],[-1,1])
    # plt.show(fig)
    if savename is not None:
        plt.savefig(savename, format='pdf', bbox_inches='tight', transparent=True, dpi=1200)
        if close:
            plt.show(fig)
            plt.close(fig)

exp=2
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if exp==2:

    data = np.load(os.path.join(ROOT_DIR,'creat_data/PGM_DGCNN_3dmatchSeen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/realdata_30.npy'),allow_pickle=True).item()
    pre_result=np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_3dmatchSeen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/eval_log_2021-01-25-13-26-36_metric.npy'),allow_pickle=True).item()

    plt.figure()  # 图片长宽和清晰度
    plt.hist(pre_result['r_mae'],50,weights=[1./len(pre_result['r_mae'])]*len(pre_result['r_mae']))
    fomatter=FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(fomatter)
    plt.title("Distribution of Rotation Error", fontdict={'family' : 'Times New Roman', 'size': 16})
    plt.ylabel('Ratio', fontdict={'family' : 'Times New Roman', 'size'   : 13})
    plt.xlabel('Rotation Error (degree)', fontdict={'family' : 'Times New Roman', 'size'   : 13})
    plt.yticks(fontproperties = 'Times New Roman', size = 13)
    plt.xticks(fontproperties = 'Times New Roman', size = 13)
    plt.savefig('fig/3dmatch/hist_r.svg', format='svg', bbox_inches='tight', transparent=True, dpi=1200)
    plt.show()


    plt.figure()  # 图片长宽和清晰度
    plt.hist(pre_result['t_mae'],50,weights=[1./len(pre_result['t_mae'])]*len(pre_result['t_mae']))
    fomatter=FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(fomatter)
    plt.title("Distribution of Translation Error", fontdict={'family' : 'Times New Roman', 'size': 16})
    plt.ylabel('Ratio', fontdict={'family' : 'Times New Roman', 'size'   : 13})
    plt.xlabel('Translation Error (m)', fontdict={'family' : 'Times New Roman', 'size'   : 13})
    plt.yticks(fontproperties = 'Times New Roman', size = 13)
    plt.xticks(fontproperties = 'Times New Roman', size = 13)
    plt.savefig('fig/3dmatch/hist_t.svg', format='svg', bbox_inches='tight', transparent=True, dpi=1200)
    plt.show()

    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1)
    #
    # ax1.hist(pre_result['r_mae'],50,weights=[1./len(pre_result['r_mae'])]*len(pre_result['r_mae']))
    # fomatter=FuncFormatter(to_percent)
    # plt.gca().yaxis.set_major_formatter(fomatter)
    # plt.title("Distribution of Rotation Error", fontdict={'family' : 'Times New Roman', 'size': 16})
    # plt.ylabel('Ratio', fontdict={'family' : 'Times New Roman', 'size'   : 13})
    # plt.xlabel('Rotation Error $^\circ$', fontdict={'family' : 'Times New Roman', 'size'   : 13})
    # plt.yticks(fontproperties = 'Times New Roman', size = 13)
    # plt.xticks(fontproperties = 'Times New Roman', size = 13)
    #
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax2.hist(pre_result['t_mae'],50,weights=[1./len(pre_result['t_mae'])]*len(pre_result['t_mae']))
    # fomatter=FuncFormatter(to_percent)
    # plt.gca().yaxis.set_major_formatter(fomatter)
    # plt.title("Distribution of Translation Error", fontdict={'family' : 'Times New Roman', 'size': 16})
    # plt.ylabel('Ratio', fontdict={'family' : 'Times New Roman', 'size'   : 13})
    # plt.xlabel('Translation Error $^\circ$', fontdict={'family' : 'Times New Roman', 'size'   : 13})
    # plt.yticks(fontproperties = 'Times New Roman', size = 13)
    # plt.xticks(fontproperties = 'Times New Roman', size = 13)
    # plt.savefig('fig/3dmatch/hist_t.pdf', format='pdf', bbox_inches='tight', transparent=True, dpi=1200)
    # plt.show()

    # print(np.where((pre_result['r_mae'] < 1) * (pre_result['t_mae'] < 0.1)))
    # # print(pre_result['r_mae'][(pre_result['r_mae'] < 1) * (pre_result['t_mae'] < 0.1)])
    #
    # pc_index = 505
    # point_size = 0.1
    # theta_2 = 90
    # theta_1 = 106
    # c0 = '#1f4f89'
    # c1 = '#b30000'
    #
    # draw_figures.show_pointcloud_2part(data['p1'][pc_index], data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/3dmatch/_before' + str(pc_index) + '.svg')
    # draw_figures.show_pointcloud_2part(se3.transform(pre_result['pre_transform'][pc_index], data['p1'][pc_index, :, :3]), data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/3dmatch/_reg' + str(pc_index) + '.svg')
    # draw_figures.show_pointcloud_2part(se3.transform(pre_result['gt_transform'][pc_index], data['raw'][pc_index, :, :3]),
    #                                    data['raw'][pc_index, :, 3:6], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/3dmatch/_reg_gt_all' + str(pc_index) + '.svg')
    # draw_figures.show_pointcloud_2part(se3.transform(pre_result['pre_transform'][pc_index], data['raw'][pc_index, :, :3]),
    #                                    data['raw'][pc_index, :, 3:6], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/3dmatch/_reg_pre_all' + str(pc_index) + '.svg')

elif exp==3:
    data = np.load(os.path.join(ROOT_DIR,'creat_data/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/Creat_dataset_log_2020-09-29-14-46-57_metric.npy'),allow_pickle=True).item()
    pre_result=np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/eval_log_2020-10-14-22-11-45_metric.npy'),allow_pickle=True).item()
    print(np.where((pre_result['r_mae'] < 1) * (pre_result['t_mae'] < 0.1)))
    # print(pre_result['r_mae'][(pre_result['r_mae'] < 1) * (pre_result['t_mae'] < 0.1)])

    index = 0
    theta_2 = 18
    theta_1 = -140
    # index=21
    # theta_2 = 89
    # theta_1 = -172

    file_num=index//4+1
    array_num=index-(file_num-1)*4
    graph_array=np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/graph/graph'+str(file_num)+'.npy'),allow_pickle=True).item()
    graph={k: graph_array[k][array_num] for k in graph_array}
    node0 = np.where(graph['A_srcz'] > 0.08)
    node1 = np.where(graph['A_tgtz'] > 0.08)
    show_2graph(se3.transform(pre_result['gt_transform'][index], data['p1'][index, :, :3]), data['p2'][index], node0, node1, theta_1=theta_1, theta_2=theta_2,savename='fig/3dmatch/vis_graph' + str(index) + '.pdf')
