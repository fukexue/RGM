pwd
echo 'start train'
echo 'clean_exp'
python train_eval_pc.py --cfg experiments/train_PGM_Seen_Clean_modelnet40_transformer.yaml
echo 'Noise_exp'
python train_eval_pc.py --cfg experiments/train_PGM_Seen_Jitter_modelnet40_transformer.yaml
echo 'crop_exp'
python train_eval_pc.py --cfg experiments/train_PGM_Seen_Crop_modelnet40_transformer.yaml
echo 'unseen_crop_exp'
python train_eval_pc.py --cfg experiments/train_PGM_Unseen_Crop_modelnet40_transformer.yaml
echo '=== ablation study exp ==='
echo 'NO AIS HAVE TRANSFORMER'
python train_eval_pc.py --cfg experiments/train_PGM_Unseen_Crop_modelnet40_NoAIS.yaml
echo 'NO TRANSFORMER HAVE AIS'
python train_eval_pc.py --cfg experiments/train_PGM_Unseen_Crop_modelnet40.yaml


echo 'realdata_exp'
python train_eval_pc.py --cfg experiments/train_PGM_Seen_3dmatch_transformer.yaml