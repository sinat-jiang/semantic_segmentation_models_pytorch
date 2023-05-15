import config
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from dataset import VOCSegDataset
from fcn import FCN
from torch.utils.data import DataLoader


def read_voc_images(root, is_train=True):
    txt_fname = '%s/ImageSets/Segmentation/%s' % (root, 'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        img_names = f.read().split()   # 拆分成一个个名字组成 list
    images, masks = [], []
    for i, img_name in tqdm(enumerate(img_names), total=len(img_names)):
        # 读入数据并且转为 RGB 的 PIL image
        images.append(Image.open(os.path.join(root, 'JPEGImages', img_name + '.jpg')).convert('RGB'))
        # 不转 RGB
        masks.append(Image.open(os.path.join(root, 'SegmentationClass', img_name + '.png')))
    return images, masks     # PIL image 0-255


def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    plt.show()
    return axes


# 构造标签矩阵
def voc_label_indices(colormap, colormap2label):
    colormap = np.array(colormap.convert("RGB")).astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx] # colormap 映射 到colormaplabel中计算的下标


def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def onehot2mask(mask):
    """
    Converts a mask (K, H, W) to (H,W)
    """
    _mask = np.argmax(mask, axis=0).astype(np.uint8)
    return _mask


def mask2onehot_test():
    """
    mask 转 one-hot 测试
    :return:
    """
    mask = np.array([[1, 1],
                     [3, 0]])
    mask = torch.from_numpy(mask).long()
    mask_onehot1 = mask2onehot(mask.numpy(), 5)  # shape = (K, H, W)
    print(mask_onehot1)


def losstest():
    import torch.nn as nn
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = VOCSegDataset(config.DATA_PATH, transform_struct=config.transforms_struct,
                                  transform_data=config.transforms_data)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.BATCH_SIZE,
                                  num_workers=config.NUM_WORKERS, pin_memory=True)
    # loss 函数测试1
    # torch.manual_seed(12)
    # outputs = torch.rand(size=(2, 6, 4, 4))  # 预测值（通道数为 class 数）
    # targets = (torch.rand(size=(2, 4, 4)) * 3).long()  # 真实 mask，只有一个通道
    # print('outputs\n', outputs)
    # # print(torch.argmax(outputs, 1))
    # print('target\n', targets)
    # loss_crit = nn.CrossEntropyLoss()
    # print(outputs.size(), targets.size())
    # outputs, targets = outputs.to(device), targets.to(device)
    # loss = loss_crit(outputs, targets)
    # print(loss)

    # loss 函数测试2
    model_fcn8x = FCN(num_class=len(config.VOC_CLASSES), in_channel=3, model_type='fcn8x')
    loss_crit = nn.CrossEntropyLoss()
    for i, (images, masks, mask_labels) in enumerate(train_dataloader):
        # masks = masks[:, 1:, :, :]
        print(images.size(), masks.size(), mask_labels.size())
        print(torch.min(masks), torch.max(masks))
        outputs = model_fcn8x(images)
        print(outputs.size())

        # 忽略特定值
        ignore_label = 255
        target_mask = masks != ignore_label         # torch.Size([2, 256, 256])
        print('target_mask:', target_mask.size())
        masks = masks[target_mask]                  # [num_pixels]

        outputs = outputs[target_mask.unsqueeze(1).repeat(1, len(config.VOC_CLASSES), 1, 1)].view(-1, len(config.VOC_CLASSES))

        print(outputs.size(), masks.size())
        ce_loss = loss_crit(outputs, masks)
        print(ce_loss)
        input()


if __name__ == '__main__':
    # data_path = r'E:\AllDateSets\CV\VOC\VOC2007\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'
    #
    # # 读取图片
    # images, masks = read_voc_images(data_path)
    # print(f'==> There are {len(images)} images for training')
    # # n = 5  # 展示几张图像
    # # imgs = images[:n] + masks[:n]  # PIL image
    # # show_images(imgs, 2, n)
    #
    # # 展示标注图片的情况
    # print(np.array(images[0]).shape)
    # print(np.array(masks[0]).shape)             # (281, 500)
    # print(np.array(masks[0])[100:110, 130:140])
    # # print(list(np.array(masks[0])))
    #
    # # colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8)  # torch.Size([16777216])
    # # for i, colormap in enumerate(config.VOC_COLORMAP):
    # #     # 每个通道的进制是256，这样可以保证每个 rgb 对应一个下标 i
    # #     colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    # #
    # # y = voc_label_indices(masks[0], colormap2label)
    # # print(y.size())
    # # # print(y[100:110, 130:140])  # 打印结果是一个int型tensor，tensor中的每个元素i表示该像素的类别是VOC_CLASSES[i]

    # mask2onehot_test()

    # print(torch.zeros([]))

    losstest()