pwd
echo '----------start-----------'
python evalpc.py --cfg experiments/test_RGM_Seen_Clean_modelnet40_transformer.yaml
python evalpc.py --cfg experiments/test_RGM_Seen_Jitter_modelnet40_transformer.yaml
python evalpc.py --cfg experiments/test_RGM_Seen_Crop_modelnet40_transformer.yaml
python evalpc.py --cfg experiments/test_RGM_Unseen_Crop_modelnet40_transformer.yaml
python evalpc.py --cfg experiments/test_RGM_Unseen_Crop_modelnet40.yaml
python evalpc.py --cfg experiments/test_RGM_Unseen_Crop_modelnet40_NoAIS.yaml