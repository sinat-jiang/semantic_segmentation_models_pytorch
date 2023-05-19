from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data.dataset import Dataset
import torch
import config
import warnings
warnings.filterwarnings("ignore")   # 忽略警告


def denormalize(x_hat):
    """逆正则化的过程，主要是保证图片再显示的时候不会出现颜色上的问题"""
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    mean, std = config.MEAN, config.STD
    # 正常的正则化计算过程：x_hat = (x-mean)/std
    # 所以有：x = x_hat*std + mean
    # 前提是必须保证 上面参与运算的 x, x_hat, mean, std 的 shape 一致
    # x: [c, h, w]
    # mean: [3] => [3, 1, 1]
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)  # unsqueeze(1) 表示在后面插入一个维度
    std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

    x = x_hat * std + mean
    return x


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


class VOCSegDataset(Dataset):
    def __init__(self, data_path, train=True, transform_struct=None, transform_data=None):
        self.data_path = data_path
        self.train = train
        self.image_crop_size = config.CROP_SIZE     # [h, w]

        # 获取所有图片存储路径
        txt_fname = '%s/ImageSets/Segmentation/%s' % (data_path, 'train.txt' if train else 'val.txt')
        with open(txt_fname, 'r') as f:
            img_names = f.read().split()  # 拆分成一个个名字组成 list
        self.images, self.masks = [], []
        for img_name in img_names:
            self.images.append(os.path.join(data_path, 'JPEGImages', img_name + '.jpg'))
            self.masks.append(os.path.join(data_path, 'SegmentationClass', img_name + '.png'))

        # 过滤尺寸小于裁剪基线的图片
        self.images, self.masks = self.filter()

        self.transform_struct = transform_struct
        self.transform_data = transform_data
        self.dataset_length = len(self.images)

        print('==> There are {} images for {}ing'.format(self.dataset_length, 'train' if train else 'val'))

    def filter(self):
        """
        过滤小于裁剪尺寸大小的图片
        :return:
        """
        images_filter, masks_filter = [], []
        for image, mask in zip(self.images, self.masks):
            w, h = Image.open(image).size   # (w, h)
            if h >= self.image_crop_size[0] and w >= self.image_crop_size[1]:
                images_filter.append(image)
                masks_filter.append(mask)
        return images_filter, masks_filter

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):

        image = self.images[index % self.dataset_length]
        mask = self.masks[index % self.dataset_length]

        image = np.array(Image.open(image).convert('RGB'))
        mask = np.array(Image.open(mask))

        if self.transform_struct:
            augmentations = self.transform_struct(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        # 对图片进行预处理标准化操作，对 label 进行处理
        if self.transform_data:
            image = self.transform_data(image=image)['image']       # 0-255 之间的像素值，直接按 mean 和 std 进行归一化

        # 忽略白边（不在这里忽略，在 loss、评价计算时再忽略）
        mask = np.array(mask).astype('int32')
        mask[mask == 255] = 0       # 直接将白边当作背景

        # 转 one-hot 编码
        # 这里由于 voc 数据集本身就是每个类别对应一中像素值，且所选择的像素值正好是从 1-20 （其中 0 代表 background），
        # 所以无需自定义映射直接用类别索引代替即可，否则需要先根据具体的像素值转为 索引，然后再转 one-hot，参考：https://www.freesion.com/article/7840535166/
        # one-hot 编码结果
        mask_label = mask2onehot(mask, len(config.VOC_CLASSES))  # shape = (K, H, W)
        # 转 tensor
        mask = torch.from_numpy(mask).long()
        mask_label = torch.from_numpy(mask_label).long()

        return image, mask, mask_label


def onehottest():
    mask = np.array([
        [0, 0, 1, 1],
        [0, 1, 2, 2],
        [1, 1, 2, 3],
        [1, 2, 3, 3]
    ])
    print(mask, mask.shape)
    mask_label = mask2onehot(mask, 4)  # shape = (K, H, W)
    print(mask_label, mask_label.shape)


if __name__ == '__main__':

    data_path = r'E:\AllDateSets\CV\VOC\VOC2007\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'

    # 随机裁剪展示
    # transform = config.transforms
    # plt.figure()
    # for i in range(5):
    #     augmentations = transform(image=np.array(images[i]), mask=np.array(masks[i]))
    #     # print(augmentations)
    #     image_trans = augmentations['image']
    #     mask_trans = augmentations['mask']
    #     plt.subplot(2, 5, i+1)
    #     plt.imshow(image_trans)
    #     plt.subplot(2, 5, 5 + i+1)
    #     plt.imshow(mask_trans)
    # plt.show()

    # 数据加载测试
    # dataset = VOCSegDataset(data_path, transform_struct=config.transforms_struct, transform_data=config.transforms_data)
    #
    # # 自动加载
    # data_loader = DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=True,
    #     num_workers=config.NUM_WORKERS,
    #     pin_memory=True
    # )
    # plt.figure()
    # count = 0
    # for step, (image, mask, mask_label) in enumerate(data_loader):
    #     print(image.size())     # torch.Size([1, 3, 256, 256])
    #     print(mask.size())      # torch.Size([1, 256, 256])
    #     print(mask_label.size())    # torch.Size([1, 21, 256, 256])
    #     print(image[:, :, 100:110, 130:140])
    #     print(mask[:, 100:110, 130:140])
    #     print(list(mask_label[0]))
    #     print('-' * 90)
    #
    #     plt.subplot(3, 5, step+1)
    #     # 反向 norm
    #     image[0] = denormalize(image[0])
    #     print(image[0][:, 100:110, 130:140])        # 0-1 之间的值
    #     plt.imshow(image[0].numpy().transpose(1, 2, 0))
    #     plt.subplot(3, 5, 5 + step + 1)
    #     plt.imshow(mask[0])
    #     plt.subplot(3, 5, 5 * 2 + step + 1)
    #     plt.imshow(mask[0][100:110, 130:140])
    #
    #     count += 1
    #     if count >= 5:
    #         break
    #
    # plt.show()

    # onehot 转换测试
    onehottest()


