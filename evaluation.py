"""
推理 & 验证
"""
from PIL import Image
from torch.utils.data import DataLoader
from dataset import VOCSegDataset
import config
import numpy as np
import torch
from utils import load_checkpoint
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2


def evaluation_val(data_path, model, device='cpu'):
    """
    验证集效果验证
    :param data_path:
    :param model:
    :return:
    """
    bs = 5
    val_dataset = VOCSegDataset(data_path, train=False, transform_sync=True, transform_img=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=bs, num_workers=0, pin_memory=True)

    model.to(device)
    model.eval()
    for i, (images, masks, mask_labels) in enumerate(val_dataloader):
        images, masks = images.to(device), masks.to(device)
        with torch.no_grad():
            outputs = model(images)

            outputs = torch.argmax(outputs, 1)
            images = val_dataset.denormalize(images)

            # 展示
            plt.figure()
            for i in range(bs):
                plt.subplot(3, bs, i + 1)
                plt.imshow(images[i].numpy().transpose(1, 2, 0))
                plt.subplot(3, bs, bs + i + 1)
                plt.imshow(outputs[i])
                plt.subplot(3, bs, bs * 2 + i + 1)
                plt.imshow(masks[i])
            plt.show()


def prediction_single(image_path, model):
    """
    对单张图片进行推理
    :param image_path:
    :return:
    """
    # 图像预处理
    image = np.array(Image.open(image_path).convert('RGB'))

    # 正则化转换
    MEAN = config.MEAN
    STD = config.STD
    transforms = A.Compose(
        [
            A.Normalize(mean=MEAN, std=STD, max_pixel_value=255),
            ToTensorV2(),
        ]
    )
    image_norm = transforms(image=image)['image']

    # 数据格式变换：(n, h, w) -> (1, n, h, w)
    image_norm = torch.from_numpy(np.array(image_norm)).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(image_norm)

        output = torch.argmax(output, 1)
        output = output[0, :, :]

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(output)
        plt.show()


if __name__ == '__main__':
    image = r'E:\AllDateSets\CV\VOC\VOC2007\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages\000033.jpg'

    # model = FCN(num_class=21)
    # checkpoint_file = './latest_epoch_weights.pth'
    # load_checkpoint(checkpoint_file=checkpoint_file, model=model, type='val')

    # model = HRnet(num_classes=21)
    # checkpoint_file = './checkpoints/hrnet/best_epoch_weights.pth'
    # load_checkpoint(checkpoint_file=checkpoint_file, model=model, type='val')

    from networks.resnet_fcn import ResNetFCN
    model = ResNetFCN(len(config.VOC_CLASSES))
    checkpoint_file = './checkpoints/resnetfcn/best_epoch_weights.pth'
    load_checkpoint(checkpoint_file=checkpoint_file, model=model, type='val')

    # 单张图片测试
    prediction_single(image, model)

    # 验证集评估
    # data_path = r'E:\AllDateSets\CV\VOC\VOC2007\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'
    # evaluation_val(data_path, model)
