import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def img_resize(img, step=15):
    # 图像缩放
    # img_part1 = img[:-2,:-2]
    h,w=img.shape
    img_copy=np.zeros_like(img)
    for i in range(0,h,step):
        for j in range(0,w,step):
            if (i+step-1)<h and (j+step-1)<w:
                if np.sum(img[i:i+step-1,j:j+step-1])>0:
                    img_copy[i:i+step-1, j:j+step-1]=1
            elif (i+step-1)>=h:
                if np.sum(img[i:-1, j:j+step-1]) > 0:
                    img_copy[i:-1, j:j+step-1] = 1
            elif (j+step-1)>=w:
                if np.sum(img[i:i+step-1, j:-1]) > 0:
                    img_copy[i:i+step-1, j:-1] = 1
    return img_copy

def covert(img, subimg):
    # 合并元图像和差异图像
    h, w = img.shape
    img_copy = np.ones((h, w, 3)) * 255
    sum_count = 0
    correct_count = 0
    error_count = 0
    for i in range(h):
        for j in range(w):
            if img[i,j]==1 and subimg[i,j]==0:
                img_copy[i ,j]=0
                sum_count=sum_count+1
                correct_count=correct_count+1
            elif subimg[i,j]==1:
                img_copy[i, j] = [255,0,0]
                sum_count = sum_count + 1
                error_count=error_count+1
    print('correct_rate:', (correct_count/sum_count)*100, '%')
    return img_copy




ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
pgm_all = np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Unseen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformer_crop/eval_log_2020-10-24-21-57-38_metric.npy'),allow_pickle=True).item()
pgm_noais = np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Unseen_NoPreW[\'xyz\', \'gxyz\']_attentiontransformernoais_crop/eval_log_2020-10-25-14-47-22_metric.npy'),allow_pickle=True).item()
pgm_notransformer = np.load(os.path.join(ROOT_DIR,'output/PGM_DGCNN_ModelNet40Unseen_NoPreW[\'xyz\', \'gxyz\']_NoAttention_crop/eval_log_2020-10-25-15-57-18_metric.npy'),allow_pickle=True).item()
#显示可挑选的图像
# print(np.where(pgm_all['r_mae']>10))
# print(np.where(pgm_noais['r_mae']>30))
# print(pgm_noais['r_mae'][pgm_noais['r_mae']>30])
# print(pgm_all['r_mae'][pgm_noais['r_mae']>30])
# a = pgm_notransformer['r_mae']>20
# b = pgm_noais['r_mae']>20
# c = pgm_notransformer['t_mae']>0.1
# d = pgm_noais['t_mae']>0.1
# e = a*b*c*d
# print(pgm_all['r_mae'][e])
# print(np.where(e))


img_gt = pgm_all['gtcorrespond_nomean']
img_pre = pgm_all['correspond_nomean']
img_pre_noais = pgm_noais['correspond_nomean']
img_pre_notransformer = pgm_notransformer['correspond_nomean']
index = 1256
h,w = img_pre[index].shape

#opencv对图像resize
# size= (int(w*0.05), int(h*0.05))
# img_pre_resize=cv2.resize(img_pre[index], size, interpolation=cv2.INTER_AREA)
# img_pre_resize[img_pre_resize>0]=1
# img_gt_resize=cv2.resize(img_gt[index], size, interpolation=cv2.INTER_AREA)
# img_gt_resize[img_gt_resize>0]=1
# img_pre_noais_resize=cv2.resize(img_pre_noais[index], size, interpolation=cv2.INTER_AREA)
# img_pre_noais_resize[img_pre_noais_resize>0]=1
# img_pre_notransformer_resize=cv2.resize(img_pre_notransformer[index], size, interpolation=cv2.INTER_AREA)
# img_pre_notransformer_resize[img_pre_notransformer_resize>0]=1


# 自己的resize函数
step=20
img_pre_resize=img_resize(img_pre[index],step)
img_gt_resize=img_resize(img_gt[index],step)
img_pre_noais_resize=img_resize(img_pre_noais[index],step)
img_pre_notransformer_resize=img_resize(img_pre_notransformer[index],step)

# 直接显示对应关系图
plt.figure()
plt.axis("off")
plt.title('fig/fig'+str(index)+'_gt.svg')
plt.imshow(img_gt_resize, cmap='gray_r')
plt.savefig('fig/fig'+str(index)+'_gt.svg', format='svg', bbox_inches='tight', transparent=True, dpi=1200)
plt.figure()
plt.axis("off")
plt.title('fig/fig'+str(index)+'_complete.svg')
plt.imshow(img_pre_resize, cmap='gray_r')
plt.savefig('fig/fig'+str(index)+'_complete.svg', format='svg', bbox_inches='tight', transparent=True, dpi=1200)
plt.figure()
plt.axis("off")
plt.title('fig/fig'+str(index)+'_noais.svg')
plt.imshow(img_pre_noais_resize, cmap='gray_r')
plt.savefig('fig/fig'+str(index)+'_noais.svg', format='svg', bbox_inches='tight', transparent=True, dpi=1200)
plt.figure()
plt.axis("off")
plt.title('fig/fig'+str(index)+'_notransformer.svg')
plt.imshow(img_pre_notransformer_resize, cmap='gray_r')
plt.savefig('fig/fig'+str(index)+'_notransformer.svg', format='svg', bbox_inches='tight', transparent=True, dpi=1200)

# 显示差异图
plt.figure()
plt.axis("off")
plt.title('fig/figsub'+str(index)+'_complete.svg')
x1 = img_pre_resize-img_gt_resize
x1[x1<0]=0
plt.imshow(x1, cmap='gray_r')
plt.savefig('fig/figsub'+str(index)+'_complete.svg', format='svg', bbox_inches='tight', transparent=True, dpi=1200)
plt.figure()
plt.axis("off")
plt.title('fig/figsub'+str(index)+'_noais.svg')
x2 = img_pre_noais_resize-img_gt_resize
x2[x2<0]=0
plt.imshow(x2, cmap='gray_r')
plt.savefig('fig/figsub'+str(index)+'_noais.svg', format='svg', bbox_inches='tight', transparent=True, dpi=1200)
plt.figure()
plt.axis("off")
plt.title('fig/figsub'+str(index)+'_notransformer.svg')
x3 = img_pre_notransformer_resize-img_gt_resize
x3[x3<0]=0
plt.imshow(x3, cmap='gray_r')
plt.savefig('fig/figsub'+str(index)+'_notransformer.svg', format='svg', bbox_inches='tight', transparent=True, dpi=1200)

# 显示合并图
img_prex = covert(img_pre_resize,x1)
img_pre_noaisx = covert(img_pre_noais_resize,x2)
img_pre_notransformerx = covert(img_pre_notransformer_resize,x3)
plt.figure()
plt.axis("off")
# plt.title('fig/fig'+str(index)+'_gt.svg')
plt.imshow(img_gt_resize, cmap='gray_r')
plt.savefig('fig/fig'+str(index)+'_gt.svg', format='svg', bbox_inches='tight', transparent=True, dpi=1200)
plt.figure()
plt.axis("off")
# plt.title('fig/fig'+str(index)+'_complete_merge.svg')
plt.imshow(img_prex, cmap='gray_r')
plt.savefig('fig/fig'+str(index)+'_complete_merge.svg', format='svg', bbox_inches='tight', transparent=True, dpi=1200)
plt.figure()
plt.axis("off")
# plt.title('fig/fig'+str(index)+'_noais_merge.svg')
plt.imshow(img_pre_noaisx, cmap='gray_r')
plt.savefig('fig/fig'+str(index)+'_noais_merge.svg', format='svg', bbox_inches='tight', transparent=True, dpi=1200)
plt.figure()
plt.axis("off")
# plt.title('fig/fig'+str(index)+'_notransformer_merge.svg')
plt.imshow(img_pre_notransformerx, cmap='gray_r')
plt.savefig('fig/fig'+str(index)+'_notransformer_merge.svg', format='svg', bbox_inches='tight', transparent=True, dpi=1200)
