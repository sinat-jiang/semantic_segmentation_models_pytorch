import torch

# training params
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 5
LR = 1e-4

INIT_EPOCH = 0
NUM_EPOCHS = 36
EVAL_EPOCH = 2
SAVE_EPOCH = 2

OPTIMIZER = 'adam'      # adam, sgd, ...
# adam params
BETAS = (0.5, 0.999)
# sgd params
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

SCHEDULER = 'mslr'      # mslr, slr, ...
# for MultiStepLR
LR_STEPS = [50, 90, 120]
LR_GAMMA_MSLR = 0.1
# for StepLR
LR_STEP_SIZE = 3         # 每多少个 epoch 降低一次学习率
LR_GAMMA_SLR = 0.1  # 学习率下降步幅

# model params
BEST_MODEL = 'best_epoch_weights.pth'
LATEST_MODEL = 'latest_epoch_weights.pth'
NUM_WORKERS = 2

# loss params
DICE_LOSS_WEIGHT = 1.0

# data params
DATA_PATH = r'E:\AllDateSets\CV\VOC\VOC2007\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'
# DATA_PATH = r'E:\AllDateSets\CV\VOC\VOCtrainval_11-May-2012\VOCdevkit\VOC2012'
# 标签其标注的类别
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# CROP_SIZE = [256, 256]
CROP_SIZE = [320, 480]  # (h, w)，这里需注意 Image.open() 打开的图片 size 结构默认是 (w, h)，对应时注意顺序

# norm params
# MEAN = [0.5, 0.5, 0.5]
# STD = [0.5, 0.5, 0.5]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


