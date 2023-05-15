import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 2
LR = 1e-4

INIT_EPOCH = 0
NUM_EPOCHS = 1000
EVAL_EPOCH = 5
SAVE_EPOCH = 5

OPTIMIZER = 'adam'      # adam, sgd, ...
# adam params
BETAS = (0.5, 0.999)
# sgd params
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

SCHEDULER = 'mslr'      # mslr, slr, ...
# for MultiStepLR
LR_STEPS = [500, 700, 900]
LR_GAMMA_MSLR = 0.1
# for StepLR
LR_STEP_SIZE = 3         # 每多少个 epoch 降低一次学习率
LR_GAMMA_SLR = 0.1  # 学习率下降步幅

# LOAD_MODEL = False      # 初始训练时指定为 False
# SAVE_MODEL = True
BEST_MODEL = './checkpoints/best_epoch_weights.pth'
LATEST_MODEL = './checkpoints/latest_epoch_weights.pth'
NUM_WORKERS = 2


DATA_PATH = r'E:\AllDateSets\CV\VOC\VOC2007\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'
# 标签其标注的类别
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# 不同颜色映射的标签值与 类别索引 的对应关系，详见：http://host.robots.ox.ac.uk/pascal/VOC/voc2007/segexamples/index.html
label2trainid = {}  # 映射关系与 VOC_CLASSES 的索引一致

CROP_SIZE = [256, 256]

# norm params
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

# 结构 transform
transforms_struct = A.Compose(
    [
        # A.Resize(width=256, height=256),
        # A.HorizontalFlip(p=0.5),
        A.RandomCrop(height=CROP_SIZE[0], width=CROP_SIZE[1]),
        # A.GaussianBlur(),
        # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        # ToTensorV2(),
    ],
    additional_targets={'mask': 'image'}
)

# 数据 transform
transforms_data = A.Compose(
    [
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=255),
        ToTensorV2(),
    ]
)

