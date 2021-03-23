pwd
echo '----------start-----------'
echo 'ModelNet40_eval:'
python eval.py --cfg experiments/test_RGM_Seen_Clean_modelnet40_transformer.yaml
python eval.py --cfg experiments/test_RGM_Seen_Jitter_modelnet40_transformer.yaml
python eval.py --cfg experiments/test_RGM_Seen_Crop_modelnet40_transformer.yaml
python eval.py --cfg experiments/test_RGM_Unseen_Crop_modelnet40_transformer.yaml
python eval.py --cfg experiments/test_RGM_Unseen_Crop_modelnet40_NoAttention.yaml
python eval.py --cfg experiments/test_RGM_Unseen_Crop_modelnet40_NoAttention_nn.yaml
python eval.py --cfg experiments/test_RGM_Unseen_Crop_modelnet40_NoAIS.yaml

echo 'ShapeNet_eval:'
python eval.py --cfg experiments/test_RGM_Seen_Clean_shapenet_transformer.yaml
python eval.py --cfg experiments/test_RGM_Seen_Jitter_shapenet_transformer.yaml
python eval.py --cfg experiments/test_RGM_Seen_Crop_shapenet_transformer.yaml