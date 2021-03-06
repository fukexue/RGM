"""Graph matching config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
from easydict import EasyDict as edict
import numpy as np

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#data3d loader settings.
__C.DATASET = edict()
__C.DATASET.BATCH_SIZE = 4
__C.DATASET.POINT_NUM = 1024
__C.DATASET.UNSEEN = False
__C.DATASET.NOISE_TYPE = 'clean'
__C.DATASET.ROT_MAG = 45.0
__C.DATASET.TRANS_MAG = 0.5
__C.DATASET.PARTIAL_P_KEEP = [0.7, 0.7]

# Pairwise data3d loader settings.
__C.PAIR = edict()
__C.PAIR.GT_GRAPH_CONSTRUCT = 'tri'     #成对的数据建立边的方案
__C.PAIR.REF_GRAPH_CONSTRUCT = 'fc'

# PGM model options
__C.PGM = edict()
__C.PGM.FEATURE_NODE_CHANNEL = 128  #节点的特征通道
__C.PGM.FEATURE_EDGE_CHANNEL = 512  #边的特征通道
__C.PGM.BS_ITER_NUM = 20    #BS双随机化参数---最大迭代次数
__C.PGM.BS_EPSILON = 1.0e-10    #BS双随机化参数---最小的eps
__C.PGM.VOTING_ALPHA = 200. #投票层参数，增大差异
__C.PGM.GNN_LAYER = 5   #嵌入层的gnn层数
__C.PGM.GNN_FEAT = 1024     #嵌入层的gnn输出特征通道数
__C.PGM.POINTER = ''     #transformer
__C.PGM.SKADDCR = False     #对于sk算法是否需要增加新的行和列
__C.PGM.SKADDCRVALUE = 0.0    #对于sk算法是否需要增加新的行和列的值
__C.PGM.USEINLIERRATE = False
__C.PGM.NORMALS = False
__C.PGM.FEATURES = ['xyz']
__C.PGM.NEIGHBORSNUM = 20
__C.PGM.USEATTEND = 'NoAttention'

# Display information
__C.VISDOM = edict()
__C.VISDOM.OPEN = False
__C.VISDOM.PORT = 8097
__C.VISDOM.SERVER = '218.244.149.221'

# Model name and dataset name
__C.MODEL_NAME = ''
__C.DATASET_NAME = ''
__C.DATASET_FULL_NAME = 'modelnet40_2048'

# Module path of module
__C.MODULE = ''

# Output path (for checkpoints, running logs and visualization results)
__C.OUTPUT_PATH = ''

# num of dataloader processes
__C.DATALOADER_NUM = __C.DATASET.BATCH_SIZE

# The step of iteration to print running statistics.
# The real step value will be the least common multiple of this value and batch_size
__C.STATISTIC_STEP = 100

# random seed used for data2d loading
__C.RANDOM_SEED = 123

# Parallel GPU indices ([0] for single GPU)
__C.GPUS = [0]

# Parallel GPU indices ([0] for single GPU)
__C.PRE_DCPWEIGHT = False


#
# Training options
#

__C.TRAIN = edict()

# # Iterations per epochs
# __C.TRAIN.EPOCH_ITERS = 7000

# Training start epoch. If not 0, will be resumed from checkpoint.
__C.TRAIN.START_EPOCH = 0   #训练开始的起点

# Total epochs
__C.TRAIN.NUM_EPOCHS = 30   #训练的epoch的数目

# Start learning rate
__C.TRAIN.LR = 0.01     #学习率
__C.TRAIN.OPTIM = ''

# Learning rate decay
__C.TRAIN.LR_DECAY = 0.1    #学习率衰减的gamma值

# Learning rate decay step (in epochs)
__C.TRAIN.LR_STEP = [10, 20]    #学习率衰减的epoch

# SGD momentum
__C.TRAIN.MOMENTUM = 0.9            #SGD中的动量值

# RobustLoss normalization
# __C.TRAIN.RLOSS_NORM = max(__C.PAIR.RESCALE)  #offset中用的参数

# Specify a class for training
# __C.TRAIN.CLASS = 'none'    #指明训练中的类别

# Loss function. Should be 'offset' or 'perm'
__C.TRAIN.LOSS_FUNC = 'perm'


#
# Evaluation options
#

__C.EVAL = edict()

# Evaluation epoch number
__C.EVAL.EPOCH = 30
__C.EVAL.ITERATION = False
__C.EVAL.CYCLE = False
__C.EVAL.ITERATION_NUM = 1

# PCK metric
# __C.EVAL.PCK_ALPHAS = [0.05, 0.10]
# __C.EVAL.PCK_L = float(max(__C.PAIR.RESCALE))  # PCK reference.

# Number of samples for testing. Stands for number of image pairs in each classes (VOC)
# __C.EVAL.SAMPLES = 1000

#
# MISC
#

# Mean and std to normalize images
# __C.NORM_MEANS = [0.485, 0.456, 0.406]
# __C.NORM_STD = [0.229, 0.224, 0.225]

#
# Experiment
#
__C.EXPERIMENT = edict()
__C.EXPERIMENT.USEPGM = True
__C.EXPERIMENT.USEREFINE = False
__C.EXPERIMENT.ICPMAXCDIST = 0.1
__C.EXPERIMENT.SHAPENET = False
__C.EXPERIMENT.USERANSAC = False
__C.EXPERIMENT.OTHERMETHODFILE = ''



def lcm(x, y):
    """
    Compute the least common multiple of x and y. This function is used for running statistics.
    """
    greater = max(x, y)
    while True:
        if (greater % x == 0) and (greater % y == 0):
            lcm = greater
            break
        greater += 1
    return lcm


def get_output_dir(model, dataset):
    """
    Return the directory where experimental artifacts are placed.
    :param model: model name
    :param dataset: dataset name
    :return: output path (checkpoint and log), visual path (visualization images)
    """
    outp_path = os.path.join('output', '{}_{}'.format(model, dataset))
    return outp_path


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d.keys()
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d.keys()
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
