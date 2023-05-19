# 修改后的网络：resnet18 作为 backbone，后接两层将输出转为语义分割所需的数据格式

import torch.nn as nn
import torch
from torchvision.models import resnet18
import numpy as np


class ResNetFCN(nn.Module):

    def __init__(self, num_class, in_channel=3, pretrained=True):
        super(ResNetFCN, self).__init__()
        self.num_class = num_class

        # 网络结构
        self.backbone = nn.Sequential(
            *list(resnet18(pretrained=pretrained).children())[:-2]
        )
        self.upsampling = nn.Sequential(
            nn.Conv2d(512, self.num_class, kernel_size=1),
            nn.ConvTranspose2d(self.num_class, self.num_class, kernel_size=64, padding=16, stride=32)  # 上采样 32 倍
        )

        # 参数初始化
        for m in self.upsampling.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=1)
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data = self.bilinear_kernel(self.num_class, self.num_class, 64)

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
        out = self.backbone(x)
        out = self.upsampling(out)
        return out


if __name__ == '__main__':
    inputs = torch.randn((2, 3, 320, 480))
    model = ResNetFCN(21)
    outputs = model(inputs)
    print(outputs.size())