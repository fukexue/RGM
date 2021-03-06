pwd
echo '-------------Creat creat_data------------'
echo 'ICP experiment'
python creat_data.py --cfg experiments/ICP_Seen_Clean_modelnet40.yaml
python creat_data.py --cfg experiments/ICP_Seen_Jitter_modelnet40.yaml
python creat_data.py --cfg experiments/ICP_Seen_Crop_modelnet40.yaml
python creat_data.py --cfg experiments/ICP_Unseen_Clean_modelnet40.yaml
python creat_data.py --cfg experiments/ICP_Unseen_Jitter_modelnet40.yaml
python creat_data.py --cfg experiments/ICP_Unseen_Crop_modelnet40.yaml
echo 'FGR experiment'
python creat_data.py --cfg experiments/FGR_Seen_Clean_modelnet40.yaml
python creat_data.py --cfg experiments/FGR_Seen_Jitter_modelnet40.yaml
python creat_data.py --cfg experiments/FGR_Seen_Crop_modelnet40.yaml
python creat_data.py --cfg experiments/FGR_Unseen_Clean_modelnet40.yaml
python creat_data.py --cfg experiments/FGR_Unseen_Jitter_modelnet40.yaml
python creat_data.py --cfg experiments/FGR_Unseen_Crop_modelnet40.yaml
echo 'DEEPGMR experiment'
python creat_data.py --cfg experiments/DEEPGMR_Seen_Clean_modelnet40.yaml
python creat_data.py --cfg experiments/DEEPGMR_Seen_Jitter_modelnet40.yaml
python creat_data.py --cfg experiments/DEEPGMR_Seen_Crop_modelnet40.yaml
python creat_data.py --cfg experiments/DEEPGMR_Unseen_Clean_modelnet40.yaml
python creat_data.py --cfg experiments/DEEPGMR_Unseen_Jitter_modelnet40.yaml
python creat_data.py --cfg experiments/DEEPGMR_Unseen_Crop_modelnet40.yaml
echo 'IDAM experiment'
python creat_data.py --cfg experiments/IDAM_Seen_Clean_modelnet40.yaml
python creat_data.py --cfg experiments/IDAM_Seen_Jitter_modelnet40.yaml
python creat_data.py --cfg experiments/IDAM_Seen_Crop_modelnet40.yaml
python creat_data.py --cfg experiments/IDAM_Unseen_Clean_modelnet40.yaml
python creat_data.py --cfg experiments/IDAM_Unseen_Jitter_modelnet40.yaml
python creat_data.py --cfg experiments/IDAM_Unseen_Crop_modelnet40.yaml
echo 'RPMNET experiment'
python creat_data.py --cfg experiments/RPMNET_Seen_Clean_modelnet40.yaml
python creat_data.py --cfg experiments/RPMNET_Seen_Jitter_modelnet40.yaml
python creat_data.py --cfg experiments/RPMNET_Seen_Crop_modelnet40.yaml
python creat_data.py --cfg experiments/RPMNET_Unseen_Clean_modelnet40.yaml
python creat_data.py --cfg experiments/RPMNET_Unseen_Clean_modelnet40au.yaml #可以用作者提供的权重，效果更好
python creat_data.py --cfg experiments/RPMNET_Unseen_Jitter_modelnet40.yaml
python creat_data.py --cfg experiments/RPMNET_Unseen_Crop_modelnet40.yaml
echo 'RGM experiment'
python creat_data.py --cfg experiments/RGM_Seen_Clean_modelnet40.yaml
python creat_data.py --cfg experiments/RGM_Seen_Jitter_modelnet40.yaml
python creat_data.py --cfg experiments/RGM_Seen_Crop_modelnet40.yaml
python creat_data.py --cfg experiments/RGM_Seen_Crop_modelnet40_transformer.yaml
python creat_data.py --cfg experiments/RGM_Seen_Cropinv_modelnet40_transformer.yaml
python creat_data.py --cfg experiments/RGM_Unseen_Clean_modelnet40.yaml
python creat_data.py --cfg experiments/RGM_Unseen_Jitter_modelnet40.yaml
python creat_data.py --cfg experiments/RGM_Unseen_Crop_modelnet40.yaml
python creat_data.py --cfg experiments/RGM_Unseen_Crop_modelnet40_transformer.yaml