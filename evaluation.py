"""
推理
"""
from PIL import Image
from torch.utils.data import DataLoader

import config
import numpy as np
import torch
from networks.fcn import FCN
from networks.hrnet import HRnet
from utils import load_checkpoint
from dataset import denormalize, onehot2mask, VOCSegDataset
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.models import resnet18


def show_images(imgs, num_rows, num_cols, scale=2):
    # a_img = np.asarray(imgs)
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    plt.show()
    return axes


def prediction_single(image_path, model):
    """
    对单张图片进行推理
    :param image_path:
    :return:
    """
    # 图像预处理
    image = np.array(Image.open(image_path).convert('RGB'))
    augmentations = config.transforms_struct(image=image)
    image = augmentations['image']
    image = config.transforms_data(image=image)['image']

    image2show = denormalize(image)

    # 数据格式变换：(n, h, w) -> (1, n, h, w)
    image = torch.from_numpy(np.array(image)).unsqueeze(0)
    print(image.size())

    model.eval()
    with torch.no_grad():
        output = model(image)
        print(output.size())
        print(list(output))

        output = torch.argmax(output, 1)
        print(output.size())
        output = output[0, :, :]
        print(output.size())
        print(list(output))

        # output = output[0, :, :, :]
        # print(output.size())

        # output = onehot2mask(output.numpy())
        # print(output.shape)

        # output = denormalize(output)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image2show.numpy().transpose(1, 2, 0))
        plt.subplot(1, 2, 2)
        plt.imshow(output)
        plt.show()


if __name__ == '__main__':
    image = r'E:\AllDateSets\CV\VOC\VOC2007\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages\000033.jpg'
    # image = r'E:\AllDateSets\CV\VOC\VOCtrainval_11-May-2012\VOCdevkit\VOC2012'

    model = FCN(num_class=21)
    checkpoint_file = './latest_epoch_weights.pth'
    load_checkpoint(checkpoint_file=checkpoint_file, model=model, type='val')

    # model = HRnet(num_classes=21)
    # checkpoint_file = './checkpoints/hrnet/best_epoch_weights.pth'
    # load_checkpoint(checkpoint_file=checkpoint_file, model=model, type='val')

    # from networks.resnet_fcn import ResNetFCN
    # model = ResNetFCN(len(config.VOC_CLASSES))
    # checkpoint_file = './resnet18_latest_model.pth'
    # load_checkpoint(checkpoint_file=checkpoint_file, model=model, type='val')

    prediction(image, model)


    # ----------------------------- 另外的测试模式 ----------------------------------
    # val_dataset = VOCSegDataset(config.DATA_PATH, transform_struct=config.transforms_struct,
    #                             transform_data=config.transforms_data, train=False)
    # val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=config.BATCH_SIZE, num_workers=0,
    #                             pin_memory=True)
    # device = 'cpu'
    # mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).to(device)
    # std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).to(device)
    # VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    #                 [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    #                 [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    #                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    #                 [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    #                 [0, 64, 128]]
    # def label2image(pred):
    #     # pred: [320,480]
    #     colormap = torch.tensor(VOC_COLORMAP, device=device, dtype=int)
    #     x = pred.long()
    #     x[x==255] = -1
    #     return (colormap[x, :]).data.cpu().numpy()
    # def visualize_model(model: nn.Module, num_images=4):
    #     was_training = model.training
    #     model.eval()
    #     images_so_far = 0
    #     n, imgs = num_images, []
    #     with torch.no_grad():
    #         for i, (inputs, labels, _) in enumerate(val_dataloader):
    #             inputs, labels = inputs.to(device), labels.to(device)  # [b,3,320,480]
    #             outputs = model(inputs)
    #             pred = torch.argmax(outputs, dim=1)  # [b,320,480]
    #             inputs_nd = (inputs * std + mean).permute(0, 2, 3, 1) * 255  # 记得要变回去哦
    #
    #             print(outputs.size())       # torch.Size([2, 21, 320, 480])
    #             print(pred.size())          # torch.Size([2, 320, 480])
    #
    #             for j in range(num_images):
    #                 images_so_far += 1
    #                 pred1 = label2image(pred[j])  # numpy.ndarray (320, 480, 3)
    #                 print(inputs_nd[j].data.int().cpu().numpy().shape, pred1.shape)
    #                 print(labels[j].shape)
    #                 print(label2image(labels[j]).shape)
    #                 imgs += [inputs_nd[j].data.int().cpu().numpy(), pred1, label2image(labels[j])]
    #                 if images_so_far == num_images:
    #                     model.train(mode=was_training)
    #                     # 我已经固定了每次只显示4张图了，大家可以自己修改
    #                     show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n)
    #                     return model.train(mode=was_training)
    #
    # # 开始验证
    # visualize_model(model)
