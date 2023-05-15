import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += (conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True))
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


class FCN8s(nn.Module):
    def __init__(self, nclass, backbone='vgg16', aux=False, pretrained_base=True, norm_layer=nn.BatchNorm2d, **kwargs):
        super(FCN8s, self).__init__()
        self.aux = aux
        if backbone == 'vgg16':
            self.pretrained = vgg16(pretrained=pretrained_base).features
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.pool3 = nn.Sequential(*self.pretrained[:17])
        self.pool4 = nn.Sequential(*self.pretrained[17:24])
        self.pool5 = nn.Sequential(*self.pretrained[24:])
        self.head = _FCNHead(512, nclass, norm_layer)
        self.score_pool3 = nn.Conv2d(256, nclass, 1)
        self.score_pool4 = nn.Conv2d(512, nclass, 1)
        if aux:
            self.auxlayer = _FCNHead(512, nclass, norm_layer)

        self.__setattr__('exclusive', ['head', 'score_pool3', 'score_pool4', 'auxlayer'] if aux else ['head', 'score_pool3', 'score_pool4'])

    def forward(self, x):
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)

        print('pool5:\n')
        print(pool5)

        outputs = []
        score_fr = self.head(pool5)

        print('score_fr\n', score_fr)

        score_pool4 = self.score_pool4(pool4)
        score_pool3 = self.score_pool3(pool3)

        upscore2 = F.interpolate(score_fr, score_pool4.size()[2:], mode='bilinear', align_corners=True)
        fuse_pool4 = upscore2 + score_pool4

        upscore_pool4 = F.interpolate(fuse_pool4, score_pool3.size()[2:], mode='bilinear', align_corners=True)
        fuse_pool3 = upscore_pool4 + score_pool3

        out = F.interpolate(fuse_pool3, x.size()[2:], mode='bilinear', align_corners=True)
        outputs.append(out)

        if self.aux:
            auxout = self.auxlayer(pool5)
            auxout = F.interpolate(auxout, x.size()[2:], mode='bilinear', align_corners=True)
            outputs.append(auxout)

        return tuple(outputs)


if __name__ == '__main__':
    fcn8x = FCN8s(21)
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
    # print(output)
    for ou in output:
        print(ou)
        print(ou.size())
        print('=' * 10)
    # print(output.size())                    # torch.Size([1, 21, 256, 256])
    # print(output[0, :, :, :])
    # print(output.long())