"""
fcn struct conference: https://blog.csdn.net/gameboxer/article/details/123493041
"""

from torchvision.models import resnet18
import torch
import torch.nn as nn
import numpy as np


class FCN(nn.Module):

    def __init__(self, num_class, in_channel=3, hidden_channels=[32, 64, 128, 256, 512], model_type='fcn8x'):
        """
        num_class 这里实际上是有效 类别数 + 1（background）
        :param num_class:
        :param in_channel:
        """
        super(FCN, self).__init__()
        self.num_class = num_class
        self.model_type = model_type

        # hidden_channels = [32, 64, 128, 256, 512]

        # first layer conv
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=hidden_channels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels[0], out_channels=hidden_channels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)   # downsampling 1/2
        )

        # second layer conv
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels[0], out_channels=hidden_channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels[1], out_channels=hidden_channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        # third layer conv
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels[1], out_channels=hidden_channels[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels[2], out_channels=hidden_channels[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        # forth layer conv
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels[2], out_channels=hidden_channels[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels[3], out_channels=hidden_channels[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        # fifth layer conv
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels[3], out_channels=hidden_channels[4], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels[4], out_channels=hidden_channels[4], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        # 第六层使用卷积层取代FC层（使用 1x1 卷积核代替 fc）
        # 分别将 layer3、layer4、layer5 的 feature map 输出 channel 变换为 num_class
        self.score_l5 = nn.Conv2d(hidden_channels[-1], self.num_class, (1, 1))
        self.score_l4 = nn.Conv2d(hidden_channels[-2], self.num_class, (1, 1))
        self.score_l3 = nn.Conv2d(hidden_channels[-3], self.num_class, (1, 1))

        # de conv & upsampling
        # size 计算公式：Wout = （Win - 1）* stride + kernal - 2 * padding + output_padding
        self.upsampling_32x = nn.ConvTranspose2d(in_channels=self.num_class, out_channels=self.num_class, kernel_size=64, stride=32, padding=16, output_padding=0)       # upsampling x32
        # 转置卷积也可以这样写:self.upsampling_2x = nn.Conv2d(in_channles, out_channels, kernel_size=4, stride=stride, padding=1, padding_mode='reflect'),
        self.upsampling_16x = nn.ConvTranspose2d(self.num_class, self.num_class, kernel_size=32, stride=16, padding=8, output_padding=0)     # upsampling x16
        self.upsampling_8x = nn.ConvTranspose2d(self.num_class, self.num_class, kernel_size=16, stride=8, padding=4, output_padding=0)       # upsampling x8
        self.upsampling_4x = nn.ConvTranspose2d(self.num_class, self.num_class, kernel_size=8, stride=4, padding=2, output_padding=0)        # upsampling x4
        self.upsampling_2x = nn.ConvTranspose2d(self.num_class, self.num_class, kernel_size=4, stride=2, padding=1, output_padding=0)        # upsampling x2

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant(m.bias, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.upsampling_32x.weight.data = self.bilinear_kernel(self.num_class, self.num_class, 64)
        self.upsampling_16x.weight.data = self.bilinear_kernel(self.num_class, self.num_class, 32)
        self.upsampling_8x.weight.data = self.bilinear_kernel(self.num_class, self.num_class, 16)
        self.upsampling_4x.weight.data = self.bilinear_kernel(self.num_class, self.num_class, 8)
        self.upsampling_2x.weight.data = self.bilinear_kernel(self.num_class, self.num_class, 4)

    def bilinear_kernel(self, in_channels, out_channels, kernel_size):
        factor = (kernel_size+1) // 2
        if kernel_size % 2 == 1:
            center = factor-1
        else:
            center = factor-0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1-abs(og[0]-center)/factor) * (1-abs(og[1]-center)/factor)
        weight = np.zeros((in_channels, out_channels, kernel_size,kernel_size), dtype='float32')
        weight[range(in_channels), range(out_channels), :, :] = filt
        weight = torch.Tensor(weight)
        weight.requires_grad = True
        return weight

    def forward(self, x):
        layer1 = self.layer1(x)             # 1/2
        layer2 = self.layer2(layer1)        # 1/4
        layer3 = self.layer3(layer2)        # 1/8
        layer4 = self.layer4(layer3)        # 1/16
        layer5 = self.layer5(layer4)        # 1/32

        sl5 = self.score_l5(layer5)           # channels -> num_class

        if self.model_type == 'fcn32x':
            out = self.upsampling_32x(sl5)
        elif self.model_type == 'fcn16x':
            sl4 = self.score_l4(layer4)
            out = sl4 + self.upsampling_2x(sl5)
            out = self.upsampling_16x(out)
        else:
            sl4 = self.score_l4(layer4)
            sl3 = self.score_l3(layer3)
            up1 = self.upsampling_2x(sl5)
            up2 = self.upsampling_2x(sl4 + up1)
            out = sl3 + up2
            out = self.upsampling_8x(out)
        return out


if __name__ == '__main__':
    fcn8x = FCN(num_class=21, in_channel=3, model_type='fcn8x')
    print(fcn8x)

    # 读取一张图片测试下
    from PIL import Image
    import os
    import numpy as np
    import config
    data_path = r'E:\AllDateSets\CV\VOC\VOC2007\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'
    txt_fname = '%s/ImageSets/Segmentation/%s' % (data_path, 'train.txt')
    with open(txt_fname, 'r') as f:
        img_names = f.read().split()  # 拆分成一个个名字组成 list
    image = os.path.join(data_path, 'JPEGImages', img_names[0] + '.jpg')
    image = Image.open(image).convert('RGB')
    image = np.array(image)
    print(image.shape)                      # (281, 500, 3)
    # crop
    image = config.transforms_struct(image=image)['image']
    print(image.shape)                      # (256, 256, 3)
    image = image.transpose(2, 0, 1)
    print(image.shape)                      # (3, 256, 256)
    image = torch.from_numpy(image).float().unsqueeze(0)
    print(image.size())                     # torch.Size([1, 3, 256, 256])

    print('-' * 90)
    output = fcn8x(image)
    print(output.size())                    # torch.Size([1, 21, 256, 256])
    print(output[0, :, :, :])
