# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

from easydict import EasyDict as edict

__C = edict()
cfg = __C

#
# Dataset Config
#
__C.DATASETS = edict()
__C.DATASETS.SHAPENET = edict()
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH = './datasets/20220330.json'
# __C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH = './datasets/PascalShapeNet.json'
#__C.DATASETS.SHAPENET.RENDERING_PATH = '/mnt/Entertainment/图像数据集/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.POINT_PATH = './datasets/ShapeNet/%s/%s/00.txt'
__C.DATASETS.SHAPENET.RENDERING_PATH = '/mnt/Entertainment/图像数据集/20211219手绘数据集/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH = '/mnt/Entertainment/图像数据集/ShapeNet/ShapeNetVox32/%s/%s/model.binvox'
__C.DATASETS.PASCAL3D = edict()
__C.DATASETS.PASCAL3D.TAXONOMY_FILE_PATH = './datasets/Pascal3D.json'
__C.DATASETS.PASCAL3D.ANNOTATION_PATH = '/home/hzxie/Datasets/PASCAL3D/Annotations/%s_imagenet/%s.mat'
__C.DATASETS.PASCAL3D.RENDERING_PATH = '/home/hzxie/Datasets/PASCAL3D/Images/%s_imagenet/%s.JPEG'
__C.DATASETS.PASCAL3D.VOXEL_PATH = '/home/hzxie/Datasets/PASCAL3D/CAD/%s/%02d.binvox'
__C.DATASETS.PIX3D = edict()
__C.DATASETS.PIX3D.TAXONOMY_FILE_PATH = './datasets/Pix3D.json'
__C.DATASETS.PIX3D.ANNOTATION_PATH = '/mnt/Entertainment/图像数据集/pix3d/pix3d.json'
__C.DATASETS.PIX3D.RENDERING_PATH = '/mnt/Entertainment/图像数据集/pix3d/img/%s/%s.%s'
__C.DATASETS.PIX3D.VOXEL_PATH = '/mnt/Entertainment/图像数据集/pix3d/model/%s/%s/%s.binvox'

#
# Dataset
#
__C.DATASET = edict()
__C.DATASET.MEAN = [0.5, 0.5, 0.5]
__C.DATASET.STD = [0.5, 0.5, 0.5]
__C.DATASET.TRAIN_DATASET = 'ShapeNet'
__C.DATASET.TEST_DATASET = 'ShapeNet'
# __C.DATASET.TEST_DATASET = 'Pascal3D'
# __C.DATASET.TEST_DATASET = 'Pix3D'

#
# Common
#
__C.CONST = edict()
__C.CONST.DEVICE = '0'
__C.CONST.RNG_SEED = 0
__C.CONST.IMG_W = 224  # Image width for input
__C.CONST.IMG_H = 224  # Image height for input
__C.CONST.N_VOX = 32
__C.CONST.BATCH_SIZE = 64
__C.CONST.N_VIEWS_RENDERING = 1  # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_W = 128  # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_H = 128  # Dummy property for Pascal 3D
__C.CONST.Z_DISTRIBUTION = 'norm'  # distribution of Z value
__C.CONST.Z_LENGTH = 512  # the length of Z value
__C.CONST.FEATURE_DIM = 64  # encoder和decoder中间的特征向量长度
__C.CONST.NPOINT = 512  # 使用pointNet结构前,取关键点数量

#
#STNet config
#
__C.STN = edict()
__C.STN.SPAN_RANGE = 0.9
__C.STN.GRID_SIZE = 4

#
# Directories
#
__C.DIR = edict()
__C.DIR.OUT_PATH = './output'
__C.DIR.RANDOM_BG_PATH = '/home/hzxie/Datasets/SUN2012/JPEGImages'

#
# Network
#
__C.NETWORK = edict()
__C.NETWORK.LEAKY_VALUE = .2
__C.NETWORK.TCONV_USE_BIAS = False
__C.NETWORK.USE_REFINER = True
__C.NETWORK.USE_MERGER = False

#
# Training
#
__C.TRAIN = edict()
__C.TRAIN.RESUME_TRAIN = False
__C.TRAIN.NUM_WORKER = 4  # number of data workers
__C.TRAIN.NUM_EPOCHES = 250
__C.TRAIN.NUM_EPOCHES_AUTOENCODER = 40
__C.TRAIN.NUM_EPOCHES_REGRESS = 60
__C.TRAIN.NUM_EPOCHES_FINETUNE = 40
__C.TRAIN.BRIGHTNESS = .4
__C.TRAIN.CONTRAST = .4
__C.TRAIN.SATURATION = .4
__C.TRAIN.NOISE_STD = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.POLICY = 'adam'  # available options: sgd, adam
__C.TRAIN.EPOCH_START_USE_REFINER = 0
__C.TRAIN.EPOCH_START_USE_MERGER = 0
__C.TRAIN.ENCODER_LEARNING_RATE = 1e-3
__C.TRAIN.DECODER_LEARNING_RATE = 1e-3
__C.TRAIN.REFINER_LEARNING_RATE = 1e-3
__C.TRAIN.MERGER_LEARNING_RATE = 1e-4
__C.TRAIN.ENCODER_LR_MILESTONES = [150]
__C.TRAIN.DECODER_LR_MILESTONES = [150]
__C.TRAIN.REFINER_LR_MILESTONES = [150]
__C.TRAIN.MERGER_LR_MILESTONES = [150]
__C.TRAIN.BETAS = (.9, .999)
__C.TRAIN.MOMENTUM = .9
__C.TRAIN.GAMMA = .5
__C.TRAIN.SAVE_FREQ = 10  # weights will be overwritten every save_freq epoch
__C.TRAIN.UPDATE_N_VIEWS_RENDERING = False

#
# Testing options
#
__C.TEST = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH = [.2, .3, .4, .5]
