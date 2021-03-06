import os
from utils import draw_figures
import numpy as np
import utils.se3 as se3

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
show_where='fig1_display'


if show_where=='fig1_display':
    #fig1_display
    PGM_SEEN_crop = np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/eval_log_2020-09-29-17-07-24_metric.npy'),allow_pickle=True).item()
    PGM_SEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/Creat_dataset_log_2020-09-29-14-46-57_metric.npy'),allow_pickle=True).item()
    print(np.where(PGM_SEEN_crop_data['label']==0))
    pc_index=21
    point_size=0.1
    c0='#00ba9e'
    c1='#f7d049'
    theta_2=104
    theta_1=-6
    draw_figures.show_pointcloud(PGM_SEEN_crop_data['p1'][pc_index],c0='#00ba9e',size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/p1_PGM_unseen_crop_reg' + str(pc_index) + '.svg', close=False)
    theta_2=132
    theta_1=-55
    draw_figures.show_pointcloud(PGM_SEEN_crop_data['p2'][pc_index],c0='#f7d049',size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/p2_PGM_unseen_crop_reg' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(PGM_SEEN_crop_data['p1'][pc_index], PGM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1,size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(se3.transform(PGM_SEEN_crop['pre_transform'][pc_index], PGM_SEEN_crop_data['p1'][pc_index, :, :3]), PGM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1,size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop_reg' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(se3.transform(PGM_SEEN_crop['gt_transform'][pc_index], PGM_SEEN_crop_data['p1'][pc_index, :, :3]), PGM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1,size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/G_unseen_crop_reg' + str(pc_index) + '.svg', close=False)


elif show_where=='fig1_display_include_inv':
    #fig1_display 包括灰色的被crop的部分
    PGM_SEEN_crop = np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/eval_log_2020-09-29-17-07-24_metric.npy'),allow_pickle=True).item()
    PGM_SEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/Creat_dataset_log_2020-09-29-14-46-57_metric.npy'),allow_pickle=True).item()
    PGM_SEEN_cropinv_data = np.load(os.path.join(ROOT_DIR,'creat_data/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_cropinv/Creat_dataset_log_2020-10-26-15-30-23_metric.npy'),allow_pickle=True).item()
    print(np.where(PGM_SEEN_crop_data['label']==0))
    pc_index=21
    point_size=0.1
    c0='#00ba9e'
    c1='#f7d049'
    c2='#777577'
    c3='#777577'
    # c3 ='#585d9c'
    theta_2=104
    theta_1=-6
    draw_figures.show_pointcloud_2part(PGM_SEEN_crop_data['p1'][pc_index], PGM_SEEN_cropinv_data['p1'][pc_index], c0=c0,
                                       c1=c2,size=point_size, theta_1=theta_1, theta_2=theta_2,
                                       savename ='fig/PGM_unseen_crop1' + str(pc_index) + '.svg', close=False)
    theta_2=50
    theta_1=-3
    draw_figures.show_pointcloud_2part(PGM_SEEN_crop_data['p2'][pc_index], PGM_SEEN_cropinv_data['p2'][pc_index], c0=c1,
                                       c1=c3, size=point_size, theta_1=theta_1, theta_2=theta_2,
                                       savename='fig/PGM_unseen_crop2' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(PGM_SEEN_crop_data['p1'][pc_index], PGM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1,size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(se3.transform(PGM_SEEN_crop['pre_transform'][pc_index], PGM_SEEN_crop_data['p1'][pc_index, :, :3]), PGM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1,size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop_reg' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(se3.transform(PGM_SEEN_crop['gt_transform'][pc_index], PGM_SEEN_crop_data['p1'][pc_index, :, :3]), PGM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1,size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/G_unseen_crop_reg' + str(pc_index) + '.svg', close=False)


elif show_where=='fig1_display_includecrop':
    #fig1_display
    PGM_SEEN_crop = np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/eval_log_2020-09-29-17-07-24_metric.npy'),allow_pickle=True).item()
    PGM_noise = np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_NoAttention_jitter/eval_log_2020-09-29-16-50-12_metric.npy'),allow_pickle=True).item()
    PGM_SEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/Creat_dataset_log_2020-09-29-14-46-57_metric.npy'),allow_pickle=True).item()
    PGM_noise_data = np.load(os.path.join(ROOT_DIR,'creat_data/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_NoAttention_jitter/Creat_dataset_log_2020-09-29-14-44-47_metric.npy'),allow_pickle=True).item()
    print(np.where(PGM_SEEN_crop_data['label']==0))
    pc_index=21
    point_size=0.1
    c0='#00ba9e'
    c1='#f7d049'
    theta_2=104
    theta_1=-6
    draw_figures.show_pointcloud(PGM_SEEN_crop_data['p1'][pc_index],c0='#00ba9e',size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/p1_PGM_unseen_crop_reg' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(PGM_SEEN_crop_data['p1'][pc_index, :, :3],PGM_noise_data['p1'][pc_index, :, :3], c0=c0, c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2,savename='fig/PGM_unseen_crop_reg' + str(pc_index) + '.svg', close=False)

    theta_2=132
    theta_1=-55
    draw_figures.show_pointcloud(PGM_SEEN_crop_data['p2'][pc_index],c0='#f7d049',size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/p2_PGM_unseen_crop_reg' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(PGM_SEEN_crop_data['p1'][pc_index], PGM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1,size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(se3.transform(PGM_SEEN_crop['pre_transform'][pc_index], PGM_SEEN_crop_data['p1'][pc_index, :, :3]), PGM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1,size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop_reg' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(se3.transform(PGM_SEEN_crop['gt_transform'][pc_index], PGM_SEEN_crop_data['p1'][pc_index, :, :3]), PGM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1,size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/G_unseen_crop_reg' + str(pc_index) + '.svg', close=False)


elif show_where=='fig1_display_zhengmian':
    #fig1_display
    PGM_SEEN_crop = np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/eval_log_2020-09-29-17-07-24_metric.npy'),allow_pickle=True).item()
    PGM_SEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/Creat_dataset_log_2020-09-29-14-46-57_metric.npy'),allow_pickle=True).item()
    print(np.where(PGM_SEEN_crop_data['label']==0))
    pc_index=21
    point_size=0.1
    c0='#00ba9e'
    c1='#f7d049'
    theta_2=129
    theta_1=-51
    draw_figures.show_pointcloud(PGM_SEEN_crop_data['p1'][pc_index],c0='#00ba9e',size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/p1_PGM_unseen_crop_reg' + str(pc_index) + '.svg', close=False)
    theta_2=97
    theta_1=-63
    draw_figures.show_pointcloud(PGM_SEEN_crop_data['p2'][pc_index],c0='#f7d049',size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/p2_PGM_unseen_crop_reg' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(PGM_SEEN_crop_data['p1'][pc_index], PGM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1,size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(se3.transform(PGM_SEEN_crop['pre_transform'][pc_index], PGM_SEEN_crop_data['p1'][pc_index, :, :3]), PGM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1,size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop_reg' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(se3.transform(PGM_SEEN_crop['gt_transform'][pc_index], PGM_SEEN_crop_data['p1'][pc_index, :, :3]), PGM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1,size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/G_unseen_crop_reg' + str(pc_index) + '.svg', close=False)


elif show_where=='clean':
    #clean  实验部分
    ICP_clean = np.load(os.path.join(ROOT_DIR,'output/ICP_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_clean/eval_log_2020-09-29-15-41-38_metric.npy'),allow_pickle=True).item()
    FGR_clean = np.load(os.path.join(ROOT_DIR,'output/FGR_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_clean/eval_log_2020-09-29-15-52-27_metric.npy'),allow_pickle=True).item()
    RPM_clean = np.load(os.path.join(ROOT_DIR,'output/RPMNET_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_clean/eval_log_2020-09-29-16-37-02_metric.npy'),allow_pickle=True).item()
    IDAM_clean = np.load(os.path.join(ROOT_DIR,'output/IDAM_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_clean/eval_log_2020-09-29-16-30-59_metric.npy'),allow_pickle=True).item()
    DEEPGMR_clean = np.load(os.path.join(ROOT_DIR,'output/DEEPGMR_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_clean/eval_log_2020-09-29-16-21-37_metric.npy'),allow_pickle=True).item()
    PGM_clean = np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_NoAttention_clean/eval_log_2020-09-29-16-46-25_metric.npy'),allow_pickle=True).item()

    ICP_clean_data = np.load(os.path.join(ROOT_DIR,'creat_data/ICP_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_clean/Creat_dataset_log_2020-09-29-15-37-47_metric.npy'),allow_pickle=True).item()
    FGR_clean_data = np.load(os.path.join(ROOT_DIR,'creat_data/FGR_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_clean/Creat_dataset_log_2020-09-29-15-26-20_metric.npy'),allow_pickle=True).item()
    DEEPGMR_clean_data = np.load(os.path.join(ROOT_DIR,'creat_data/DEEPGMR_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_clean/Creat_dataset_log_2020-09-29-14-08-44_metric.npy'),allow_pickle=True).item()
    IDAM_clean_data = np.load(os.path.join(ROOT_DIR,'creat_data/IDAM_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_clean/Creat_dataset_log_2020-09-29-14-18-34_metric.npy'),allow_pickle=True).item()
    PGM_clean_data = np.load(os.path.join(ROOT_DIR,'creat_data/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_NoAttention_clean/Creat_dataset_log_2020-09-29-14-40-28_metric.npy'),allow_pickle=True).item()
    RPM_clean_data = np.load(os.path.join(ROOT_DIR,'creat_data/RPMNET_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_clean/Creat_dataset_log_2020-09-29-14-28-23_metric.npy'),allow_pickle=True).item()
    print(np.where(PGM_clean['r_mae']>10))
    print(np.where(PGM_clean['t_mae'] > 0.1))
    print(PGM_clean['r_mae'][RPM_clean['r_mae'] > 10])
    print(PGM_clean['r_mae'][RPM_clean['t_mae'] >0.1])
    print(np.where(RPM_clean['r_mae']>10))
    print(np.where(RPM_clean['t_mae'] >0.1))
    print(RPM_clean['r_mae'][RPM_clean['t_mae'] > 0.1])

    pc_index=1645
    point_size=0.1
    # c0='#00ba9e'
    # c1='#f7d049'
    c0 = '#1f4f89'
    c1 = '#b30000'
    theta_2=17
    theta_1=66
    draw_figures.show_pointcloud_2part(ICP_clean_data['p1'][pc_index], ICP_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/ICP_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(FGR_clean_data['p1'][pc_index], FGR_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/FGR_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(DEEPGMR_clean_data['p1'][pc_index], DEEPGMR_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/DEEPGMR_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(IDAM_clean_data['p1'][pc_index], IDAM_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/IDAM_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(RPM_clean_data['p1'][pc_index], RPM_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/RPM_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(PGM_clean_data['p1'][pc_index], PGM_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop' + str(pc_index) + '.svg')

    draw_figures.show_pointcloud_2part(se3.transform(ICP_clean['pre_transform'][pc_index], ICP_clean_data['p1'][pc_index, :, :3]), ICP_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/ICP_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(FGR_clean['pre_transform'][pc_index], FGR_clean_data['p1'][pc_index, :, :3]), FGR_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/FGR_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(DEEPGMR_clean['pre_transform'][pc_index], DEEPGMR_clean_data['p1'][pc_index, :, :3]), DEEPGMR_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/DEEPGMR_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(IDAM_clean['pre_transform'][pc_index], IDAM_clean_data['p1'][pc_index, :, :3]), IDAM_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/IDAM_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(RPM_clean['pre_transform'][pc_index], RPM_clean_data['p1'][pc_index, :, :3]), RPM_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/RPM_unseen_crop_reg' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(se3.transform(PGM_clean['pre_transform'][pc_index], PGM_clean_data['p1'][pc_index, :, :3]), PGM_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(PGM_clean['gt_transform'][pc_index], PGM_clean_data['p1'][pc_index, :, :3]), PGM_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/G_unseen_crop_reg' + str(pc_index) + '.svg', close=False)


elif show_where=='clean2':
    #clean
    ICP_clean = np.load(os.path.join(ROOT_DIR,'output/ICP_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_clean/eval_log_2020-09-29-15-41-38_metric.npy'),allow_pickle=True).item()
    FGR_clean = np.load(os.path.join(ROOT_DIR,'output/FGR_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_clean/eval_log_2020-09-29-15-52-27_metric.npy'),allow_pickle=True).item()
    RPM_clean = np.load(os.path.join(ROOT_DIR,'output/RPMNET_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_clean/eval_log_2020-09-29-16-37-02_metric.npy'),allow_pickle=True).item()
    IDAM_clean = np.load(os.path.join(ROOT_DIR,'output/IDAM_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_clean/eval_log_2020-09-29-16-30-59_metric.npy'),allow_pickle=True).item()
    DEEPGMR_clean = np.load(os.path.join(ROOT_DIR,'output/DEEPGMR_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_clean/eval_log_2020-09-29-16-21-37_metric.npy'),allow_pickle=True).item()
    PGM_clean = np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_clean/eval_log_2020-10-13-18-49-28_metric.npy'),allow_pickle=True).item()

    ICP_clean_data = np.load(os.path.join(ROOT_DIR,'creat_data/ICP_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_clean/Creat_dataset_log_2020-09-29-15-37-47_metric.npy'),allow_pickle=True).item()
    FGR_clean_data = np.load(os.path.join(ROOT_DIR,'creat_data/FGR_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_clean/Creat_dataset_log_2020-09-29-15-26-20_metric.npy'),allow_pickle=True).item()
    DEEPGMR_clean_data = np.load(os.path.join(ROOT_DIR,'creat_data/DEEPGMR_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_clean/Creat_dataset_log_2020-09-29-14-08-44_metric.npy'),allow_pickle=True).item()
    IDAM_clean_data = np.load(os.path.join(ROOT_DIR,'creat_data/IDAM_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_clean/Creat_dataset_log_2020-09-29-14-18-34_metric.npy'),allow_pickle=True).item()
    PGM_clean_data = np.load(os.path.join(ROOT_DIR,'creat_data/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_NoAttention_clean/Creat_dataset_log_2020-09-29-14-40-28_metric.npy'),allow_pickle=True).item()
    RPM_clean_data = np.load(os.path.join(ROOT_DIR,'creat_data/RPMNET_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_clean/Creat_dataset_log_2020-09-29-14-28-23_metric.npy'),allow_pickle=True).item()
    # print(np.where(PGM_clean['label'] ==26))
    print(np.where(PGM_clean['r_mae']> 10))
    print(np.where(PGM_clean['t_mae'] > 0.1))
    print(PGM_clean['r_mae'][RPM_clean['r_mae'] > 5])
    print(PGM_clean['r_mae'][RPM_clean['t_mae'] >0.05])
    print(np.where(ICP_clean['r_mae']> 30))
    print(ICP_clean['r_mae'][ICP_clean['r_mae']> 30])
    print(ICP_clean['t_mae'][ICP_clean['r_mae']> 30])
    print(np.where(ICP_clean['t_mae'] >0.3))
    print(ICP_clean['r_mae'][ICP_clean['t_mae'] > 0.3])
    print(ICP_clean['t_mae'][ICP_clean['t_mae'] > 0.3])

    pc_index=913
    point_size=0.1
    # c0='#00ba9e'
    # c1='#f7d049'
    c0 = '#1f4f89'
    c1 = '#b30000'
    theta_2=23
    theta_1=90
    draw_figures.show_pointcloud_2part(ICP_clean_data['p1'][pc_index], ICP_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/ICP_unseen_crop' + str(pc_index) + '.svg')
    # draw_figures.show_pointcloud_2part(FGR_clean_data['p1'][pc_index], FGR_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/FGR_unseen_crop' + str(pc_index) + '.svg')
    # draw_figures.show_pointcloud_2part(DEEPGMR_clean_data['p1'][pc_index], DEEPGMR_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/DEEPGMR_unseen_crop' + str(pc_index) + '.svg')
    # draw_figures.show_pointcloud_2part(IDAM_clean_data['p1'][pc_index], IDAM_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/IDAM_unseen_crop' + str(pc_index) + '.svg')
    # draw_figures.show_pointcloud_2part(RPM_clean_data['p1'][pc_index], RPM_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/RPM_unseen_crop' + str(pc_index) + '.svg')
    # draw_figures.show_pointcloud_2part(PGM_clean_data['p1'][pc_index], PGM_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop' + str(pc_index) + '.svg')

    draw_figures.show_pointcloud_2part(se3.transform(ICP_clean['pre_transform'][pc_index], ICP_clean_data['p1'][pc_index, :, :3]), ICP_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/ICP_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(FGR_clean['pre_transform'][pc_index], FGR_clean_data['p1'][pc_index, :, :3]), FGR_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/FGR_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(DEEPGMR_clean['pre_transform'][pc_index], DEEPGMR_clean_data['p1'][pc_index, :, :3]), DEEPGMR_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/DEEPGMR_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(IDAM_clean['pre_transform'][pc_index], IDAM_clean_data['p1'][pc_index, :, :3]), IDAM_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/IDAM_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(RPM_clean['pre_transform'][pc_index], RPM_clean_data['p1'][pc_index, :, :3]), RPM_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/RPM_unseen_crop_reg' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(se3.transform(PGM_clean['pre_transform'][pc_index], PGM_clean_data['p1'][pc_index, :, :3]), PGM_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(PGM_clean['gt_transform'][pc_index], PGM_clean_data['p1'][pc_index, :, :3]), PGM_clean_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/G_unseen_crop_reg' + str(pc_index) + '.svg', close=False)


elif show_where=='noise':
    #noise
    ICP_noise = np.load(os.path.join(ROOT_DIR,'output/ICP_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_jitter/eval_log_2020-09-29-15-43-47_metric.npy'),allow_pickle=True).item()
    FGR_noise = np.load(os.path.join(ROOT_DIR,'output/FGR_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_jitter/eval_log_2020-09-29-15-24-48_metric.npy'),allow_pickle=True).item()
    RPM_noise = np.load(os.path.join(ROOT_DIR,'output/RPMNET_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_jitter/eval_log_2020-09-29-16-37-34_metric.npy'),allow_pickle=True).item()
    IDAM_noise = np.load(os.path.join(ROOT_DIR,'output/IDAM_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_jitter/eval_log_2020-09-29-16-31-30_metric.npy'),allow_pickle=True).item()
    DEEPGMR_noise = np.load(os.path.join(ROOT_DIR,'output/DEEPGMR_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_jitter/eval_log_2020-09-29-16-22-09_metric.npy'),allow_pickle=True).item()
    PGM_noise = np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_NoAttention_jitter/eval_log_2020-09-29-16-50-12_metric.npy'),allow_pickle=True).item()

    ICP_noise_data = np.load(os.path.join(ROOT_DIR,'creat_data/ICP_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_jitter/Creat_dataset_log_2020-09-29-15-38-13_metric.npy'),allow_pickle=True).item()
    FGR_noise_data = np.load(os.path.join(ROOT_DIR,'creat_data/FGR_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_jitter/Creat_dataset_log_2020-09-29-14-03-11_metric.npy'),allow_pickle=True).item()
    DEEPGMR_noise_data = np.load(os.path.join(ROOT_DIR,'creat_data/DEEPGMR_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_jitter/Creat_dataset_log_2020-09-29-14-12-59_metric.npy'),allow_pickle=True).item()
    IDAM_noise_data = np.load(os.path.join(ROOT_DIR,'creat_data/IDAM_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_jitter/Creat_dataset_log_2020-09-29-14-22-52_metric.npy'),allow_pickle=True).item()
    PGM_noise_data = np.load(os.path.join(ROOT_DIR,'creat_data/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_NoAttention_jitter/Creat_dataset_log_2020-09-29-14-44-47_metric.npy'),allow_pickle=True).item()
    RPM_noise_data = np.load(os.path.join(ROOT_DIR,'creat_data/RPMNET_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_jitter/Creat_dataset_log_2020-09-29-14-32-41_metric.npy'),allow_pickle=True).item()
    print(np.where(PGM_noise['r_mae']>10))
    print(np.where(RPM_noise['r_mae']>10))

    pc_index=2196
    point_size=0.1
    c0='#00ba9e'
    c1='#f7d049'
    theta_2=-87
    theta_1=159
    draw_figures.show_pointcloud_2part(ICP_noise_data['p1'][pc_index], ICP_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/ICP_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(FGR_noise_data['p1'][pc_index], FGR_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/FGR_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(DEEPGMR_noise_data['p1'][pc_index], DEEPGMR_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/DEEPGMR_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(IDAM_noise_data['p1'][pc_index], IDAM_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/IDAM_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(RPM_noise_data['p1'][pc_index], RPM_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/RPM_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(PGM_noise_data['p1'][pc_index], PGM_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop' + str(pc_index) + '.svg')

    draw_figures.show_pointcloud_2part(se3.transform(ICP_noise['pre_transform'][pc_index], ICP_noise_data['p1'][pc_index, :, :3]), ICP_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/ICP_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(FGR_noise['pre_transform'][pc_index], FGR_noise_data['p1'][pc_index, :, :3]), FGR_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/FGR_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(DEEPGMR_noise['pre_transform'][pc_index], DEEPGMR_noise_data['p1'][pc_index, :, :3]), DEEPGMR_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/DEEPGMR_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(IDAM_noise['pre_transform'][pc_index], IDAM_noise_data['p1'][pc_index, :, :3]), IDAM_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/IDAM_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(RPM_noise['pre_transform'][pc_index], RPM_noise_data['p1'][pc_index, :, :3]), RPM_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/RPM_unseen_crop_reg' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(se3.transform(PGM_noise['pre_transform'][pc_index], PGM_noise_data['p1'][pc_index, :, :3]), PGM_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(PGM_noise['gt_transform'][pc_index], PGM_noise_data['p1'][pc_index, :, :3]), PGM_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/G_unseen_crop_reg' + str(pc_index) + '.svg', close=False)


elif show_where=='noise2':
    #noise
    ICP_noise = np.load(os.path.join(ROOT_DIR,'output/ICP_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_jitter/eval_log_2020-09-29-15-43-47_metric.npy'),allow_pickle=True).item()
    FGR_noise = np.load(os.path.join(ROOT_DIR,'output/FGR_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_jitter/eval_log_2020-09-29-15-24-48_metric.npy'),allow_pickle=True).item()
    RPM_noise = np.load(os.path.join(ROOT_DIR,'output/RPMNET_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_jitter/eval_log_2020-09-29-16-37-34_metric.npy'),allow_pickle=True).item()
    IDAM_noise = np.load(os.path.join(ROOT_DIR,'output/IDAM_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_jitter/eval_log_2020-09-29-16-31-30_metric.npy'),allow_pickle=True).item()
    DEEPGMR_noise = np.load(os.path.join(ROOT_DIR,'output/DEEPGMR_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_jitter/eval_log_2020-09-29-16-22-09_metric.npy'),allow_pickle=True).item()
    PGM_noise = np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_jitter/eval_log_2020-10-14-10-38-35_metric.npy'),allow_pickle=True).item()

    ICP_noise_data = np.load(os.path.join(ROOT_DIR,'creat_data/ICP_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_jitter/Creat_dataset_log_2020-09-29-15-38-13_metric.npy'),allow_pickle=True).item()
    FGR_noise_data = np.load(os.path.join(ROOT_DIR,'creat_data/FGR_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_jitter/Creat_dataset_log_2020-09-29-14-03-11_metric.npy'),allow_pickle=True).item()
    DEEPGMR_noise_data = np.load(os.path.join(ROOT_DIR,'creat_data/DEEPGMR_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_jitter/Creat_dataset_log_2020-09-29-14-12-59_metric.npy'),allow_pickle=True).item()
    IDAM_noise_data = np.load(os.path.join(ROOT_DIR,'creat_data/IDAM_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_jitter/Creat_dataset_log_2020-09-29-14-22-52_metric.npy'),allow_pickle=True).item()
    PGM_noise_data = np.load(os.path.join(ROOT_DIR,'creat_data/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_NoAttention_jitter/Creat_dataset_log_2020-09-29-14-44-47_metric.npy'),allow_pickle=True).item()
    RPM_noise_data = np.load(os.path.join(ROOT_DIR,'creat_data/RPMNET_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_jitter/Creat_dataset_log_2020-09-29-14-32-41_metric.npy'),allow_pickle=True).item()
    print(np.where(PGM_noise['r_mae']>10))
    print(np.where(PGM_noise['t_mae'] > 0.1))
    print(PGM_noise['r_mae'][RPM_noise['r_mae'] > 10])
    print(PGM_noise['r_mae'][RPM_noise['t_mae'] >0.1])
    print('FGR')
    print(np.where(FGR_noise['r_mae']> 10))
    print(FGR_noise['r_mae'][FGR_noise['r_mae']> 10])
    print(FGR_noise['t_mae'][FGR_noise['r_mae']> 10])
    print(np.where(FGR_noise['t_mae'] >0.3))
    print(FGR_noise['r_mae'][FGR_noise['t_mae'] > 0.3])
    print(FGR_noise['t_mae'][FGR_noise['t_mae'] > 0.3])
    print('IDAM')
    print(np.where(IDAM_noise['r_mae']> 10))
    print(IDAM_noise['r_mae'][IDAM_noise['r_mae']> 10])
    print(IDAM_noise['t_mae'][IDAM_noise['r_mae']> 10])
    print(np.where(IDAM_noise['t_mae'] >0.1))
    print(IDAM_noise['r_mae'][IDAM_noise['t_mae'] > 0.1])
    print(IDAM_noise['t_mae'][IDAM_noise['t_mae'] > 0.1])
    print('DEEPGMR')
    print(np.where(DEEPGMR_noise['r_mae']> 5))
    print(DEEPGMR_noise['r_mae'][DEEPGMR_noise['r_mae']> 5])
    print(DEEPGMR_noise['t_mae'][DEEPGMR_noise['r_mae']> 5])
    print(np.where(DEEPGMR_noise['t_mae'] >0.03))
    print(DEEPGMR_noise['r_mae'][DEEPGMR_noise['t_mae'] > 0.03])
    print(DEEPGMR_noise['t_mae'][DEEPGMR_noise['t_mae'] > 0.03])


    pc_index=2152
    theta_2=90
    theta_1=0
    point_size=0.1
    # c0='#00ba9e'
    # c1='#f7d049'
    c0 = '#1f4f89'
    c1 = '#b30000'
    # pc_index=1145
    # theta_2=-92
    # theta_1=-176
    draw_figures.show_pointcloud_2part(ICP_noise_data['p1'][pc_index], ICP_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/ICP_unseen_crop' + str(pc_index) + '.svg')
    # draw_figures.show_pointcloud_2part(FGR_noise_data['p1'][pc_index], FGR_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/FGR_unseen_crop' + str(pc_index) + '.svg')
    # draw_figures.show_pointcloud_2part(DEEPGMR_noise_data['p1'][pc_index], DEEPGMR_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/DEEPGMR_unseen_crop' + str(pc_index) + '.svg')
    # draw_figures.show_pointcloud_2part(IDAM_noise_data['p1'][pc_index], IDAM_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/IDAM_unseen_crop' + str(pc_index) + '.svg')
    # draw_figures.show_pointcloud_2part(RPM_noise_data['p1'][pc_index], RPM_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/RPM_unseen_crop' + str(pc_index) + '.svg')
    # draw_figures.show_pointcloud_2part(PGM_noise_data['p1'][pc_index], PGM_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop' + str(pc_index) + '.svg')

    draw_figures.show_pointcloud_2part(se3.transform(ICP_noise['pre_transform'][pc_index], ICP_noise_data['p1'][pc_index, :, :3]), ICP_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/ICP_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(FGR_noise['pre_transform'][pc_index], FGR_noise_data['p1'][pc_index, :, :3]), FGR_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/FGR_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(DEEPGMR_noise['pre_transform'][pc_index], DEEPGMR_noise_data['p1'][pc_index, :, :3]), DEEPGMR_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/DEEPGMR_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(IDAM_noise['pre_transform'][pc_index], IDAM_noise_data['p1'][pc_index, :, :3]), IDAM_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/IDAM_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(RPM_noise['pre_transform'][pc_index], RPM_noise_data['p1'][pc_index, :, :3]), RPM_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/RPM_unseen_crop_reg' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(se3.transform(PGM_noise['pre_transform'][pc_index], PGM_noise_data['p1'][pc_index, :, :3]), PGM_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(PGM_noise['gt_transform'][pc_index], PGM_noise_data['p1'][pc_index, :, :3]), PGM_noise_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/G_unseen_crop_reg' + str(pc_index) + '.svg', close=False)


elif show_where=='seen_partial5':
    # 测试时五次迭代的实验结果   部分重叠看的见类
    #seen_partial
    ICP_SEEN_crop = np.load(os.path.join(ROOT_DIR,'output/ICP_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_crop/eval_log_2020-09-29-15-46-53_metric.npy'),allow_pickle=True).item()
    FGR_SEEN_crop = np.load(os.path.join(ROOT_DIR,'output/FGR_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_crop/eval_log_2020-09-29-16-07-18_metric.npy'),allow_pickle=True).item()
    RPM_SEEN_crop = np.load(os.path.join(ROOT_DIR,'output/RPMNET_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/eval_log_2020-09-29-16-38-58_metric.npy'),allow_pickle=True).item()
    IDAM_SEEN_crop = np.load(os.path.join(ROOT_DIR,'output/IDAM_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/eval_log_2020-09-29-16-32-57_metric.npy'),allow_pickle=True).item()
    DEEPGMR_SEEN_crop = np.load(os.path.join(ROOT_DIR,'output/DEEPGMR_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/eval_log_2020-09-29-16-23-35_metric.npy'),allow_pickle=True).item()
    PGM_SEEN_crop = np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/eval_log_2020-09-29-17-07-24_metric.npy'),allow_pickle=True).item()

    ICP_SEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/ICP_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_crop/Creat_dataset_log_2020-09-29-15-39-36_metric.npy'),allow_pickle=True).item()
    FGR_SEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/FGR_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_crop/Creat_dataset_log_2020-09-29-14-04-29_metric.npy'),allow_pickle=True).item()
    DEEPGMR_SEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/DEEPGMR_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/Creat_dataset_log_2020-09-29-14-14-20_metric.npy'),allow_pickle=True).item()
    IDAM_SEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/IDAM_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/Creat_dataset_log_2020-09-29-14-24-09_metric.npy'),allow_pickle=True).item()
    RPM_SEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/RPMNET_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/Creat_dataset_log_2020-09-29-14-33-59_metric.npy'),allow_pickle=True).item()
    PGM_SEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/Creat_dataset_log_2020-09-29-14-46-57_metric.npy'),allow_pickle=True).item()
    pc_index=21
    point_size=0.1
    theta_2=0
    theta_1=0
    c0='#00ba9e'
    c1='#f7d049'
    draw_figures.show_pointcloud_2part(ICP_SEEN_crop_data['p1'][pc_index], ICP_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/ICP_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(FGR_SEEN_crop_data['p1'][pc_index], FGR_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/FGR_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(DEEPGMR_SEEN_crop_data['p1'][pc_index], DEEPGMR_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/DEEPGMR_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(IDAM_SEEN_crop_data['p1'][pc_index], IDAM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/IDAM_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(RPM_SEEN_crop_data['p1'][pc_index], RPM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/RPM_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(PGM_SEEN_crop_data['p1'][pc_index], PGM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop' + str(pc_index) + '.svg', close=False)

    draw_figures.show_pointcloud_2part(se3.transform(ICP_SEEN_crop['pre_transform'][pc_index], ICP_SEEN_crop_data['p1'][pc_index, :, :3]), ICP_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/ICP_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(FGR_SEEN_crop['pre_transform'][pc_index], FGR_SEEN_crop_data['p1'][pc_index, :, :3]), FGR_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/FGR_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(DEEPGMR_SEEN_crop['pre_transform'][pc_index], DEEPGMR_SEEN_crop_data['p1'][pc_index, :, :3]), DEEPGMR_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/DEEPGMR_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(IDAM_SEEN_crop['pre_transform'][pc_index], IDAM_SEEN_crop_data['p1'][pc_index, :, :3]), IDAM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/IDAM_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(RPM_SEEN_crop['pre_transform'][pc_index], RPM_SEEN_crop_data['p1'][pc_index, :, :3]), RPM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/RPM_unseen_crop_reg' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(se3.transform(PGM_SEEN_crop['pre_transform'][pc_index], PGM_SEEN_crop_data['p1'][pc_index, :, :3]), PGM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop_reg' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(se3.transform(PGM_SEEN_crop['gt_transform'][pc_index], PGM_SEEN_crop_data['p1'][pc_index, :, :3]), PGM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/G_unseen_crop_reg' + str(pc_index) + '.svg', close=False)


elif show_where=='seen_partial2':
    # 测试时五次迭代的实验结果   部分重叠看的见类
    #seen_partial
    ICP_SEEN_crop = np.load(os.path.join(ROOT_DIR,'output/ICP_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_crop/eval_log_2020-09-29-15-46-53_metric.npy'),allow_pickle=True).item()
    FGR_SEEN_crop = np.load(os.path.join(ROOT_DIR,'output/FGR_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_crop/eval_log_2020-09-29-16-07-18_metric.npy'),allow_pickle=True).item()
    RPM_SEEN_crop = np.load(os.path.join(ROOT_DIR,'output/RPMNET_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/eval_log_2020-09-29-16-38-58_metric.npy'),allow_pickle=True).item()
    IDAM_SEEN_crop = np.load(os.path.join(ROOT_DIR,'output/IDAM_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/eval_log_2020-09-29-16-32-57_metric.npy'),allow_pickle=True).item()
    DEEPGMR_SEEN_crop = np.load(os.path.join(ROOT_DIR,'output/DEEPGMR_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/eval_log_2020-09-29-16-23-35_metric.npy'),allow_pickle=True).item()
    PGM_SEEN_crop = np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/eval_log_2020-10-14-22-11-45_metric.npy'),allow_pickle=True).item()

    ICP_SEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/ICP_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_crop/Creat_dataset_log_2020-09-29-15-39-36_metric.npy'),allow_pickle=True).item()
    FGR_SEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/FGR_ModelNet40Seen_NoPreW[\'xyz\']_NoAttention_crop/Creat_dataset_log_2020-09-29-14-04-29_metric.npy'),allow_pickle=True).item()
    DEEPGMR_SEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/DEEPGMR_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/Creat_dataset_log_2020-09-29-14-14-20_metric.npy'),allow_pickle=True).item()
    IDAM_SEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/IDAM_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/Creat_dataset_log_2020-09-29-14-24-09_metric.npy'),allow_pickle=True).item()
    RPM_SEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/RPMNET_ModelNet40Seen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/Creat_dataset_log_2020-09-29-14-33-59_metric.npy'),allow_pickle=True).item()
    PGM_SEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/PGM_DGCNN_ModelNet40Seen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/Creat_dataset_log_2020-09-29-14-46-57_metric.npy'),allow_pickle=True).item()
    print(np.where(PGM_SEEN_crop['r_mae']>10))
    print(np.where(PGM_SEEN_crop['t_mae'] > 0.1))
    print(PGM_SEEN_crop['r_mae'][RPM_SEEN_crop['r_mae'] > 10])
    print(PGM_SEEN_crop['r_mae'][RPM_SEEN_crop['t_mae'] >0.1])
    print(np.where(RPM_SEEN_crop['r_mae']>10))
    print(np.where(RPM_SEEN_crop['t_mae'] >0.1))
    print(RPM_SEEN_crop['r_mae'][RPM_SEEN_crop['t_mae'] > 0.1])


    pc_index=2313
    point_size=0.1
    theta_2=-109
    theta_1=-21
    # c0='#00ba9e'
    # c1='#f7d049'
    c0 = '#1f4f89'
    c1 = '#b30000'
    # pc_index=3
    # theta_2=-77
    # theta_1=79
    # pc_index = 857
    # theta_2 = -92
    # theta_1 = 105
    # pc_index=1588
    # theta_2=-87
    # theta_1=150
    # pc_index=2313
    # theta_2=-109
    # theta_1=-21
    draw_figures.show_pointcloud_2part(ICP_SEEN_crop_data['p1'][pc_index], ICP_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/ICP_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(FGR_SEEN_crop_data['p1'][pc_index], FGR_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/FGR_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(DEEPGMR_SEEN_crop_data['p1'][pc_index], DEEPGMR_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/DEEPGMR_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(IDAM_SEEN_crop_data['p1'][pc_index], IDAM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/IDAM_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(RPM_SEEN_crop_data['p1'][pc_index], RPM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/RPM_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(PGM_SEEN_crop_data['p1'][pc_index], PGM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop' + str(pc_index) + '.svg', close=False)

    draw_figures.show_pointcloud_2part(se3.transform(ICP_SEEN_crop['pre_transform'][pc_index], ICP_SEEN_crop_data['p1'][pc_index, :, :3]), ICP_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/ICP_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(FGR_SEEN_crop['pre_transform'][pc_index], FGR_SEEN_crop_data['p1'][pc_index, :, :3]), FGR_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/FGR_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(DEEPGMR_SEEN_crop['pre_transform'][pc_index], DEEPGMR_SEEN_crop_data['p1'][pc_index, :, :3]), DEEPGMR_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/DEEPGMR_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(IDAM_SEEN_crop['pre_transform'][pc_index], IDAM_SEEN_crop_data['p1'][pc_index, :, :3]), IDAM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/IDAM_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(RPM_SEEN_crop['pre_transform'][pc_index], RPM_SEEN_crop_data['p1'][pc_index, :, :3]), RPM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/RPM_unseen_crop_reg' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(se3.transform(PGM_SEEN_crop['pre_transform'][pc_index], PGM_SEEN_crop_data['p1'][pc_index, :, :3]), PGM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop_reg' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(se3.transform(PGM_SEEN_crop['gt_transform'][pc_index], PGM_SEEN_crop_data['p1'][pc_index, :, :3]), PGM_SEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/G_unseen_crop_reg' + str(pc_index) + '.svg', close=False)


elif show_where=='unseen_partial5':
    # 测试时五次迭代的实验结果     部分重叠看不见的类
    #unseen_partial
    print('可以更改要显示的点云的index')
    ICP_UNSEEN_crop = np.load(os.path.join(ROOT_DIR,'output/ICP_ModelNet40Unseen_NoPreW[\'xyz\']_NoAttention_crop/eval_log_2020-09-29-16-03-22_metric.npy'),allow_pickle=True).item()
    FGR_UNSEEN_crop = np.load(os.path.join(ROOT_DIR,'output/FGR_ModelNet40Unseen_NoPreW[\'xyz\']_NoAttention_crop/eval_log_2020-09-29-16-15-53_metric.npy'),allow_pickle=True).item()
    RPM_UNSEEN_crop = np.load(os.path.join(ROOT_DIR,'output/RPMNET_ModelNet40Unseen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/eval_log_2020-09-29-16-41-32_metric.npy'),allow_pickle=True).item()
    IDAM_UNSEEN_crop = np.load(os.path.join(ROOT_DIR,'output/IDAM_ModelNet40Unseen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/eval_log_2020-09-29-16-35-09_metric.npy'),allow_pickle=True).item()
    DEEPGMR_UNSEEN_crop = np.load(os.path.join(ROOT_DIR,'output/DEEPGMR_ModelNet40Unseen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/eval_log_2020-09-29-16-27-48_metric.npy'),allow_pickle=True).item()
    PGM_UNSEEN_crop = np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Unseen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/eval_log_2020-09-29-18-07-40_metric.npy'),allow_pickle=True).item()

    ICP_UNSEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/ICP_ModelNet40Unseen_NoPreW[\'xyz\']_NoAttention_crop/Creat_dataset_log_2020-09-29-15-25-43_metric.npy'),allow_pickle=True).item()
    FGR_UNSEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/FGR_ModelNet40Unseen_NoPreW[\'xyz\']_NoAttention_crop/Creat_dataset_log_2020-09-29-14-08-16_metric.npy'),allow_pickle=True).item()
    DEEPGMR_UNSEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/DEEPGMR_ModelNet40Unseen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/Creat_dataset_log_2020-09-29-14-18-06_metric.npy'),allow_pickle=True).item()
    IDAM_UNSEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/IDAM_ModelNet40Unseen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/Creat_dataset_log_2020-09-29-14-27-56_metric.npy'),allow_pickle=True).item()
    PGM_UNSEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/PGM_DGCNN_ModelNet40Unseen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/Creat_dataset_log_2020-09-29-14-51-11_metric.npy'),allow_pickle=True).item()
    RPM_UNSEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/RPMNET_ModelNet40Unseen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/Creat_dataset_log_2020-09-29-14-40-00_metric.npy'),allow_pickle=True).item()
    print(np.where(PGM_UNSEEN_crop['r_mae']>10))
    print(np.where(RPM_UNSEEN_crop['r_mae']>10))

    pc_index=1
    theta_2=90
    theta_1=0
    point_size=0.1
    c0='#00ba9e'
    c1='#f7d049'
    # pc_index=28
    # theta_2=91
    # theta_1=81
    # pc_index=514
    # theta_2=87
    # theta_1=-26
    # pc_index=1113
    # theta_2=-97
    # theta_1=-99
    draw_figures.show_pointcloud_2part(ICP_UNSEEN_crop_data['p1'][pc_index], ICP_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/ICP_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(FGR_UNSEEN_crop_data['p1'][pc_index], FGR_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/FGR_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(DEEPGMR_UNSEEN_crop_data['p1'][pc_index], DEEPGMR_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/DEEPGMR_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(IDAM_UNSEEN_crop_data['p1'][pc_index], IDAM_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/IDAM_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(RPM_UNSEEN_crop_data['p1'][pc_index], RPM_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/RPM_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(PGM_UNSEEN_crop_data['p1'][pc_index], PGM_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop' + str(pc_index) + '.svg')

    draw_figures.show_pointcloud_2part(se3.transform(ICP_UNSEEN_crop['pre_transform'][pc_index], ICP_UNSEEN_crop_data['p1'][pc_index, :, :3]), ICP_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/ICP_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(FGR_UNSEEN_crop['pre_transform'][pc_index], FGR_UNSEEN_crop_data['p1'][pc_index, :, :3]), FGR_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/FGR_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(DEEPGMR_UNSEEN_crop['pre_transform'][pc_index], DEEPGMR_UNSEEN_crop_data['p1'][pc_index, :, :3]), DEEPGMR_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/DEEPGMR_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(IDAM_UNSEEN_crop['pre_transform'][pc_index], IDAM_UNSEEN_crop_data['p1'][pc_index, :, :3]), IDAM_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/IDAM_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(RPM_UNSEEN_crop['pre_transform'][pc_index], RPM_UNSEEN_crop_data['p1'][pc_index, :, :3]), RPM_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/RPM_unseen_crop_reg' + str(pc_index) + '.svg', close=False)
    draw_figures.show_pointcloud_2part(se3.transform(PGM_UNSEEN_crop['pre_transform'][pc_index], PGM_UNSEEN_crop_data['p1'][pc_index, :, :3]), PGM_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(PGM_UNSEEN_crop['gt_transform'][pc_index], PGM_UNSEEN_crop_data['p1'][pc_index, :, :3]), PGM_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/G_unseen_crop_reg' + str(pc_index) + '.svg', close=False)


elif show_where=='unseen_partial2':
    # 测试时2次迭代的实验结果    部分重叠看不见的类
    #unseen_partial
    print('可以更改要显示的点云的index')
    ICP_UNSEEN_crop = np.load(os.path.join(ROOT_DIR,'output/ICP_ModelNet40Unseen_NoPreW[\'xyz\']_NoAttention_crop/eval_log_2020-09-29-16-03-22_metric.npy'),allow_pickle=True).item()
    FGR_UNSEEN_crop = np.load(os.path.join(ROOT_DIR,'output/FGR_ModelNet40Unseen_NoPreW[\'xyz\']_NoAttention_crop/eval_log_2020-09-29-16-15-53_metric.npy'),allow_pickle=True).item()
    RPM_UNSEEN_crop = np.load(os.path.join(ROOT_DIR,'output/RPMNET_ModelNet40Unseen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/eval_log_2020-09-29-16-41-32_metric.npy'),allow_pickle=True).item()
    IDAM_UNSEEN_crop = np.load(os.path.join(ROOT_DIR,'output/IDAM_ModelNet40Unseen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/eval_log_2020-09-29-16-35-09_metric.npy'),allow_pickle=True).item()
    DEEPGMR_UNSEEN_crop = np.load(os.path.join(ROOT_DIR,'output/DEEPGMR_ModelNet40Unseen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/eval_log_2020-09-29-16-27-48_metric.npy'),allow_pickle=True).item()
    PGM_UNSEEN_crop = np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Unseen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/eval_log_2020-10-13-11-06-51_metric.npy'),allow_pickle=True).item()

    ICP_UNSEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/ICP_ModelNet40Unseen_NoPreW[\'xyz\']_NoAttention_crop/Creat_dataset_log_2020-09-29-15-25-43_metric.npy'),allow_pickle=True).item()
    FGR_UNSEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/FGR_ModelNet40Unseen_NoPreW[\'xyz\']_NoAttention_crop/Creat_dataset_log_2020-09-29-14-08-16_metric.npy'),allow_pickle=True).item()
    DEEPGMR_UNSEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/DEEPGMR_ModelNet40Unseen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/Creat_dataset_log_2020-09-29-14-18-06_metric.npy'),allow_pickle=True).item()
    IDAM_UNSEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/IDAM_ModelNet40Unseen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/Creat_dataset_log_2020-09-29-14-27-56_metric.npy'),allow_pickle=True).item()
    PGM_UNSEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/PGM_DGCNN_ModelNet40Unseen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/Creat_dataset_log_2020-09-29-14-51-11_metric.npy'),allow_pickle=True).item()
    RPM_UNSEEN_crop_data = np.load(os.path.join(ROOT_DIR,'creat_data/RPMNET_ModelNet40Unseen_NoPreW[\'xyz\', \'normal\']_NoAttention_crop/Creat_dataset_log_2020-09-29-14-40-00_metric.npy'),allow_pickle=True).item()
    print(np.where(PGM_UNSEEN_crop['r_mae']>10))
    print(np.where(PGM_UNSEEN_crop['t_mae'] > 0.1))
    print(PGM_UNSEEN_crop['r_mae'][RPM_UNSEEN_crop['r_mae'] > 10])
    print(PGM_UNSEEN_crop['r_mae'][RPM_UNSEEN_crop['t_mae'] >0.1])
    print(np.where(RPM_UNSEEN_crop['r_mae']>10))
    print(np.where(RPM_UNSEEN_crop['t_mae'] >0.1))
    print(RPM_UNSEEN_crop['r_mae'][RPM_UNSEEN_crop['t_mae'] > 0.1])

    point_size=0.1
    # c0='#00ba9e'
    # c1='#f7d049'
    c0 = '#1f4f89'
    c1 = '#b30000'
    pc_index=510
    theta_2=19
    theta_1=2
    # pc_index=28
    # theta_2=91
    # theta_1=81
    # pc_index=368
    # theta_2=75
    # theta_1=28
    # pc_index=1210
    # theta_2=86
    # theta_1=-139
    # pc_index=1060
    # theta_2=86
    # theta_1=-139
    draw_figures.show_pointcloud_2part(ICP_UNSEEN_crop_data['p1'][pc_index], ICP_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/ICP_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(FGR_UNSEEN_crop_data['p1'][pc_index], FGR_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/FGR_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(DEEPGMR_UNSEEN_crop_data['p1'][pc_index], DEEPGMR_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/DEEPGMR_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(IDAM_UNSEEN_crop_data['p1'][pc_index], IDAM_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/IDAM_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(RPM_UNSEEN_crop_data['p1'][pc_index], RPM_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/RPM_unseen_crop' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(PGM_UNSEEN_crop_data['p1'][pc_index], PGM_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop' + str(pc_index) + '.svg')

    draw_figures.show_pointcloud_2part(se3.transform(ICP_UNSEEN_crop['pre_transform'][pc_index], ICP_UNSEEN_crop_data['p1'][pc_index, :, :3]), ICP_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/ICP_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(FGR_UNSEEN_crop['pre_transform'][pc_index], FGR_UNSEEN_crop_data['p1'][pc_index, :, :3]), FGR_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/FGR_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(DEEPGMR_UNSEEN_crop['pre_transform'][pc_index], DEEPGMR_UNSEEN_crop_data['p1'][pc_index, :, :3]), DEEPGMR_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/DEEPGMR_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(IDAM_UNSEEN_crop['pre_transform'][pc_index], IDAM_UNSEEN_crop_data['p1'][pc_index, :, :3]), IDAM_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/IDAM_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(RPM_UNSEEN_crop['pre_transform'][pc_index], RPM_UNSEEN_crop_data['p1'][pc_index, :, :3]), RPM_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/RPM_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(PGM_UNSEEN_crop['pre_transform'][pc_index], PGM_UNSEEN_crop_data['p1'][pc_index, :, :3]), PGM_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/PGM_unseen_crop_reg' + str(pc_index) + '.svg')
    draw_figures.show_pointcloud_2part(se3.transform(PGM_UNSEEN_crop['gt_transform'][pc_index], PGM_UNSEEN_crop_data['p1'][pc_index, :, :3]), PGM_UNSEEN_crop_data['p2'][pc_index], c0=c0,c1=c1, size=point_size, theta_1=theta_1, theta_2=theta_2, savename ='fig/G_unseen_crop_reg' + str(pc_index) + '.svg')


else:
    print('no show')