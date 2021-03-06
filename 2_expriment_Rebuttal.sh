pwd
echo '------start rebuttal experiments----------'

# ShapeNet exp
echo 'ShapeNet experiment'
echo 'PGM PART'
python evalpc.py --cfg experiments/PGM_Seen_Clean_shapenet_transformer.yaml
python evalpc.py --cfg experiments/PGM_Seen_Jitter_shapenet_transformer.yaml
python evalpc.py --cfg experiments/PGM_Seen_Crop_shapenet_transformer.yaml
echo 'RPMNET PRAT'
python evalpc_othermethod.py --cfg experiments/RPMNET_Seen_Clean_shapenet.yaml
python evalpc_othermethod.py --cfg experiments/RPMNET_Seen_Jitter_shapenet.yaml
python evalpc_othermethod.py --cfg experiments/RPMNET_Seen_Crop_shapenet.yaml

##关于角度的鲁棒性实验
#echo 'degree test exp'
##之前的网络训练结果不好 结果没有删除
#python evalpc.py --cfg experiments/PGM_Seen_Clean_modelnet40_transformer_360fdeg.yaml
#python evalpc.py --cfg experiments/PGM_Seen_Jitter_modelnet40_transformer_360deg.yaml
#python evalpc.py --cfg experiments/PGM_Seen_Crop_modelnet40_transformer_360deg.yaml

#echo 'degree train exp'
#python train_eval_pc.py --cfg experiments/PGM_Seen_Clean_modelnet40_transformer_360fdeg.yaml
#python train_eval_pc.py --cfg experiments/PGM_Seen_Jitter_modelnet40_transformer_360deg.yaml
#python train_eval_pc.py --cfg experiments/PGM_Seen_Crop_modelnet40_transformer_360deg.yaml
