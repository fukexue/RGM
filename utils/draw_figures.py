import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d.axes3d import Axes3D


def show_pointcloud_batch(pc, size=10):
    if type(pc) == torch.Tensor:
        pc = pc.detach().cpu().numpy()
    if pc.shape[1]==3:
        pc = pc.transpose(0,2,1)
    B,N,C = pc.shape
    fig = plt.figure()
    for i in range(B):
        ax = fig.add_subplot(2, int(B/2), i+1, projection='3d')
        ax.scatter(pc[i, :, 0], pc[i, :, 1], pc[i, :, 2], s=size, alpha=0.5)
        plt.axis("off")
        ax.view_init(0,0)


def show_pointcloud_batch_2part(pc0, pc1, size=10, c0='r', c1='b'):
    if type(pc0) == torch.Tensor:
        pc0 = pc0.detach().cpu().numpy()
    if pc0.shape[1]==3:
        pc0 = pc0.transpose(0,2,1)
    if type(pc1) == torch.Tensor:
        pc1 = pc1.detach().cpu().numpy()
    if pc1.shape[1]==3:
        pc1 = pc1.transpose(0,2,1)
    B,N,C = pc0.shape
    fig = plt.figure()
    for i in range(B):
        ax = fig.add_subplot(2, int(B/2), i+1, projection='3d')
        ax.scatter(pc0[i, :, 0], pc0[i, :, 1], pc0[i, :, 2], s=size, alpha=0.5, c=c0)
        ax.scatter(pc1[i, :, 0], pc1[i, :, 1], pc1[i, :, 2], s=size, alpha=0.5, c=c1)
        plt.axis("off")
        ax.view_init(0,0)


def show_pointcloud(pc, size=10, theta_1=0, theta_2=0, c0='#b30000',savename=None, close=True):
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
    if savename is not None:
        plt.savefig(savename, format='svg', bbox_inches='tight', transparent=True, dpi=600)
        if close:
            plt.close(fig)
    # legend = ax.legend()
    # frame = legend.get_frame()
    # frame.set_alpha(1)
    # frame.set_facecolor('none')


def show_pointcloud_2part(pc0, pc1, size=10, theta_1=0, theta_2=0, c0='#1f4f89', c1='#b30000', savename=None, close=True):
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
    plt.axis("off")
    ax.view_init(theta_1, theta_2)
    ax.auto_scale_xyz([-1,1],[-1,1],[-1,1])
    if savename is not None:
        plt.savefig(savename, format='svg', bbox_inches='tight', transparent=True, dpi=1200)
        if close:
            plt.show(fig)
            plt.close(fig)


def rotate_one_axis(X, axis, rotation_angle, return_angle=False):
    """
    Apply random rotation about one axis
    Input:
        x: pointcloud data, [N, 3]
        rotation_angle: theta
        axis: axis to do rotation about
    Return:
        A rotated shape
    """
    # rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    if axis == 'x':
        R_x = [[1, 0, 0], [0, cosval, -sinval], [0, sinval, cosval]]
        X = np.matmul(X, R_x)
    elif axis == 'y':
        R_y = [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        X = np.matmul(X, R_y)
    else:
        R_z = [[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]]
        X = np.matmul(X, R_z)
    if return_angle:
        return X.astype('float32'), rotation_angle
    else:
        return X.astype('float32')


def norm_pc_01(pc):
    # pc.shape==Nx3
    return (pc - pc.min(0))/(pc.max(0)-pc.min(0))


def plotly_show_pointcloud(pc, size=4, c='#b30000', eye=None, savename=None):
    ## pc.shape = Nx3
    ## eye: camera position; eg: eye=[1, 1, 0.1]
    ## savename shuold have no suffix
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Scatter3d(x=pc[:,0],
                                       y=pc[:,1],
                                       z=pc[:,2],
                                       mode='markers',
                                       marker=dict(
                                           symbol='circle',
                                           size=size,
                                           color=c,
                                           opacity=1,
                                       ))],
                    layout=go.Layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        template="none",
                        showlegend=False,
                        scene=dict(
                            xaxis=dict(
                                title='',
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                            ),
                            yaxis=dict(
                                title='',
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                            ),
                            zaxis=dict(
                                title='',
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                            )
                        ),
                    ))
    if eye is not None:
        fig.update_layout(
            scene_camera=dict(
                eye=dict(x=eye[0],y=eye[1],z=eye[2])
            )
        )
    fig.show()
    if savename is not None:
        fig.write_html(savename+'.html')
        fig.write_image(savename+'.svg')


def plotly_show_2pointcloud(pc1, pc2, size=4, c0='#1f4f89', c1='#b30000', eye=None, rangex=[-0.9,0.9], rangey=[-0.9,0.9], rangez=[-0.9,0.9], savename=None):
    ## pc.shape = Nx3
    ## eye: camera position; eg: eye=[1, 1, 0.1]
    ## savename shuold have no suffix
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Scatter3d(x=pc1[:,0],
                                       y=pc1[:,1],
                                       z=pc1[:,2],
                                       mode='markers',
                                       marker=dict(
                                           symbol='circle',
                                           size=size,
                                           color=c0,
                                           opacity=0.5,
                                       )),
                          go.Scatter3d(x=pc2[:, 0],
                                       y=pc2[:, 1],
                                       z=pc2[:, 2],
                                       mode='markers',
                                       marker=dict(
                                           symbol='circle',
                                           size=size,
                                           color=c1,
                                           opacity=0.5,
                                       ))],
                    layout=go.Layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        template="none",
                        showlegend=False,
                        scene=dict(
                            xaxis=dict(
                                title='',
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                                range=rangex
                            ),
                            yaxis=dict(
                                title='',
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                                range=rangey
                            ),
                            zaxis=dict(
                                title='',
                                showgrid=False,
                                zeroline=False,
                                showticklabels=False,
                                range=rangez
                            )
                        ),
                    ))
    if eye is not None:
        fig.update_layout(
            scene_camera=dict(
                eye=dict(x=eye[0],y=eye[1],z=eye[2])
            )
        )
    fig.show()
    if savename is not None:
        fig.write_html(savename+'.html')
        fig.write_image(savename+'.svg', scale=10)


def show_pc_vox_grid(pc, voxels, pc_size = 1, vox_alpha=0.5, grid_width=1, ax=None):
    if type(pc) == torch.Tensor:
        pc = pc.cpu().detach().numpy()
    if type(voxels) == torch.Tensor:
        voxels = voxels.cpu().detach().numpy()
    K = voxels.shape[0]
    # for i in voxels.shape[0]:
    #     for j in voxels.shape[1]:
    #         for k in voxels.shape[2]:
    #             if pc * np.array([voxels.shape[0],voxels.shape[1],voxels.shape[2]])
    pc_global = pc * np.array([voxels.shape[0],voxels.shape[1],voxels.shape[2]])
    pc_grid = np.array(np.where(np.ones([K, K, K])>0)).transpose() + np.array([0.5,0.5,0.5])
    import utliz
    dis_pc2grid = utliz.EuclideanDistances(pc_global, pc_grid)
    idx_on = np.array(np.where(dis_pc2grid.min(0).reshape(K, K, K)<0.5))
    voxels[idx_on[0], idx_on[1], idx_on[2]] = 1
    on_voxels = np.where(voxels>=1)
    x_min = on_voxels[0].min()-1
    x_max = on_voxels[0].max()+1
    y_min = on_voxels[1].min()-1
    y_max = on_voxels[1].max()+1
    z_min = on_voxels[2].min()-1
    z_max = on_voxels[2].max()+1
    pc_local = pc#-np.array([x_min, y_min, z_min])
    print(pc_local.min(0))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(pc_global[:,0], pc_global[:,1], pc_global[:,2],s=pc_size)
    # ax.voxels(voxels[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1], alpha=vox_alpha)
    ax.voxels(voxels, alpha=vox_alpha)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(on_voxels[0]+0.5, on_voxels[1]+0.5, on_voxels[2]+0.5, s=pc_size*3, c='r')

    # add grid line
    from itertools import product
    r = range(0, K+1)
    # line width
    lw = grid_width
    # line color
    lc = '#919191'
    # get all position in a 2d plane
    coordinates = product(r, repeat=2)
    # draw line parallel to x axis
    x = (r[0], r[-1])
    for y, z in coordinates:
        ax.plot3D(x, [y, y], [z, z], color=lc, lw=lw)
        ax.plot3D([y, y], x, [z, z], color=lc, lw=lw)
        ax.plot3D([y, y], [z, z], x, color=lc, lw=lw)
    plt.axis('off')
    return np.array(on_voxels).transpose() + np.array([0.5, 0.5, 0.5])

# show_pc_vox_grid(norm_pc_01(src_data[1].transpose()), np.zeros([10,10,10]), pc_size=5, vox_alpha=0.2,  grid_width=0.1 )

if __name__ == '__main__':
    ### code bellow used to save PC as .np file
    ### should be run in console while running train_nn_DGCNN_maintask_noscannet.py file
    #
    # pc_destoryed_d, pc_destoryed_left = model_des(pc1_1, pc1_2, alpha = args.Gan_alpha, return_not_cat=True)
    # pc_destoryed = torch.cat([pc_destoryed_d, pc_destoryed_left], dim=2)
    # pc_destoryed_feat, feat_cat = model_pn(pc_destoryed, out_feat_cat=True)  # ,grads = grads, name = 'Gan')
    # pc_destoryed_cat_feat = pc_destoryed_feat.unsqueeze(dim=2)
    # pc_destoryed_cat_feat = torch.cat(
    #     (feat_cat.squeeze(dim=3), pc_destoryed_cat_feat.repeat(1, 1, src_data.shape[2])), dim=1)
    # pc_pred = model_restore(pc_destoryed_cat_feat)
    # show_pointcloud_batch(src_data[0:8])
    # show_pointcloud_batch_2part(pc_destoryed_d, pc_destoryed_left)
    # results_save_dir = "./results_figures_data/9_17/local_0/" # replace local_0 to local_1 or global to save PC in other conditions
    # np.save(os.path.join(results_save_dir, "src_data"), src_data.detach().cpu().numpy())
    # np.save(os.path.join(results_save_dir, "pc_destoryed_d"), pc_destoryed_d.detach().cpu().numpy())
    # np.save(os.path.join(results_save_dir, "pc_destoryed_left"), pc_destoryed_left.detach().cpu().numpy())
    # np.save(os.path.join(results_save_dir, "pc_pred"), pc_pred.detach().cpu().numpy())

    ### code bellow used to show PC
    ### should be run in console while running train_nn_DGCNN_maintask_noscannet.py file
    ### show data in batch (deformed part in different color)
    # pc_destoryed = np.concatenate([pc_destoryed_d, pc_destoryed_left], axis=2)
    # show_pointcloud_batch(src_data[0:8])
    # show_pointcloud_batch(pc_destoryed[0:8])
    # show_pointcloud_batch_2part(pc_destoryed_d, pc_destoryed_left)
    # show_pointcloud_batch(pc_pred[0:8])


    for des_type in ["local_0", "local_1", "global"]:
        # replace local_0 to local_1 or global to load PC other conditions
        # load saved pc data(generated and saved by code above)
        results_save_dir = os.path.join("./results_figures_data/9_17/", des_type)
        src_data = np.load(os.path.join(results_save_dir, "src_data"+".npy"))
        pc_destoryed_d = np.load(os.path.join(results_save_dir, "pc_destoryed_d"+".npy"))
        pc_destoryed_left = np.load(os.path.join(results_save_dir, "pc_destoryed_left"+".npy"))
        pc_pred = np.load(os.path.join(results_save_dir, "pc_pred"+".npy"))

        num_pc = [1]
        theta_1 = 17
        theta_2 = -150
        show_pointcloud(src_data[num_pc], theta_1=theta_1, theta_2=theta_2,
                        savename=os.path.join(results_save_dir, "chair_view0.png"))
        show_pointcloud_2part(pc_destoryed_d[num_pc], pc_destoryed_left[num_pc],
                              theta_1=theta_1, theta_2=theta_2, savename=os.path.join(results_save_dir, "chair_view1.png"))

        # draw deform pc
        # show_pointcloud_2part(src_data_deform[num_pc] * (src_mask[num_pc]),
        #                       src_data_deform[num_pc] * (1 - src_mask[num_pc]),
        #                       theta_1=theta_1, theta_2=theta_2,
        #                       savename=os.path.join(results_save_dir, "chair_defRec.png"))