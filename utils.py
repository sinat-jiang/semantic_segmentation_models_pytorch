"""
评价函数参考自：https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/d37d2a17221d2681ad454958cf06a1065e9b1f7f/core/utils/score.py#L69
"""
import torch
import torch.nn as nn


def save_checkpoint(model, optimizer, lr_scheduler, filename='my_checkpoint.pth.tar'):
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer=None, lr_scheduler=None, lr=None, device='cpu', type='train'):
    print('==> Loading checkpoint')
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    if type == 'train':
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        # # if wen don't do this then it will just have learning rate of old checkpoint and it will lead to many hours of debugging
        # for param_group in optimizer.param_group:
        #     param_group['lr'] = lr

        print('-' * 90)
        print('optimizer lr:')
        for param_group in optimizer.param_group:
            print(param_group['lr'])
        print('-' * 90)

    # return model


def batch_pix_accuracy(output, target, ignore_label=None):
    """
    PixAcc
    :param output: (b, c, h, w)
    :param target: mask (b, h, w)
    :return:
    """
    # 忽略白边（貌似不手动忽略，而是通过 + 1 的方式巧妙的避免计算也行）
    if ignore_label is not None:
        target[target == ignore_label] = -1

    # 加 1 是为了方便计算
    predict = torch.argmax(output, 1) + 1   # (b, channel=class, h, w) -> (b, h, w)
    target = target.long() + 1              # (b, h, w)

    pixel_labeled = torch.sum(target > 0).item()
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass, ignore_label=None):
    """
    mIoU
    :param output:
    :param target:
    :param nclass:
    :param ignore_label:
    :return:
    """
    # 忽略白边（貌似不手动忽略，而是通过 + 1 的方式巧妙的避免计算也行）
    if ignore_label is not None:
        target[target == ignore_label] = -1

    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass

    predict = torch.argmax(output, 1) + 1      # (b, channel=class, h, w) -> (b, h, w)
    target = target.float() + 1                # (b, h, w)

    predict = predict.float() * (target > 0).float()        # 这一步感觉啥都没做啊，就转了个 float
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    # 通过直方图函数来计算每个类别的像素个数，histc 返回一个长度为 nclass 的 vector，其中元素依次表示每个类别的像素个数
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)        # 相交区域
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)              # 预测区域
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)                # 真实标签区域
    area_union = area_pred + area_lab - area_inter                                      # 相并区域
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()


class CrossEntropy2d(nn.Module):
    def __init__(self, nclass, ignore_label=None):
        super().__init__()
        self.nclass = nclass
        self.ignore_label = ignore_label

    def forward(self, preds, masks):
        """
        :param preds: outputs of segmentation network, [batch, num_class, height, width]
        :param masks: the masks of the correspond images, [batch, height, width]
        :return: entropy loss
        """
        if self.ignore_label is not None:
            # 忽略特定值
            target_mask = masks != self.ignore_label    # 筛选蒙版，torch.Size([bs, h, w])
            masks = masks[target_mask]                  # [num_pixels]，筛选后的所有 masks 像素值排成一列

            # 将 target_mask 由 (bs, h, w) -> (bs, nclass, h, w)，其实就是在新增 dim=1 维度上复制 nclass 份，再根据 target_mask 对 preds 的像素进行筛选
            preds = preds[target_mask.unsqueeze(1).repeat(1, self.nclass, 1, 1)].view(-1, self.nclass)

        # CrossEntropyLoss() 会自动的执行 softmax 和 one-hot 转换
        ce_loss = nn.CrossEntropyLoss()(preds, masks)

        return ce_loss


if __name__ == '__main__':
    # pa、miou 计算测试
    torch.manual_seed(12)
    outputs = torch.rand(size=(2, 3, 4, 4))             # 预测值（通道数为 class 数）
    target = (torch.rand(size=(2, 4, 4)) * 3).long()    # 真实 mask，只有一个通道
    print('outputs\n', outputs)
    # print(torch.argmax(outputs, 1))
    print('target\n', target)

    print('-' * 90)
    correct, labeled = batch_pix_accuracy(outputs, target)
    print(correct, labeled)

    print('-' * 90)
    inter, union = batch_intersection_union(outputs, target, 3)
    print(inter, union)

    print('=' * 90)
    nclass = 3
    total_inter = torch.zeros(nclass)
    total_union = torch.zeros(nclass)
    total_correct = 0
    total_label = 0

    total_correct += correct
    total_label += labeled
    # if total_inter.device != inter.device:
    #     total_inter = total_inter.to(inter.device)
    #     total_union = total_union.to(union.device)
    total_inter += inter
    total_union += union

    # 计算 pa、miou
    pixAcc = 1.0 * total_correct / (2.220446049250313e-16 + total_label)  # remove np.spacing(1)
    IoU = 1.0 * total_inter / (2.220446049250313e-16 + total_union)
    mIoU = IoU.mean().item()
    print(pixAcc)
    print(IoU)
    print(mIoU)

