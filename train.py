"""
training
"""
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import argparse
import sys
import config
from networks.fcn import FCN
from networks.resnet_fcn import ResNetFCN
from dataset import VOCSegDataset
from utils import load_checkpoint, save_checkpoint, batch_intersection_union, batch_pix_accuracy, CrossEntropy2d, DiceLoss2d


# base for path
# cur_path = os.path.abspath(os.path.dirname(__file__))
# root_path = os.path.split(cur_path)[0]
# sys.path.append(root_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='fcn',
                        choices=['fcn32s', 'fcn16s', 'fcn8s', 'fcn', 'psp', 'deeplabv3',
                                 'deeplabv3_plus', 'danet', 'denseaspp', 'bisenet', 'encnet',
                                 'dunet', 'icnet', 'enet', 'ocnet', 'psanet', 'cgnet', 'espnet',
                                 'lednet', 'dfanet', 'swnet'],
                        help='model name (default: fcn32s)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet18', 'resnet50', 'resnet101', 'resnet152',
                                 'densenet121', 'densenet161', 'densenet169', 'densenet201'],
                        help='backbone name (default: vgg16)')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        choices=['pascal_voc', 'pascal_aug', 'ade20k', 'citys', 'sbu'],
                        help='dataset name (default: pascal_voc)')
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--workers', '-j', type=int, default=4,
                        metavar='N', help='dataloader threads')
    # training hyper params
    parser.add_argument('--jpu', action='store_true', default=False,
                        help='JPU')
    parser.add_argument('--use-ohem', type=bool, default=False,
                        help='OHEM Loss for cityscapes dataset')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--warmup-iters', type=int, default=0,
                        help='warmup iters')
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3,
                        help='lr = warmup_factor * lr')
    parser.add_argument('--warmup-method', type=str, default='linear',
                        help='method of warmup')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='~/.torch/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'pascal_aug': 80,
            'pascal_voc': 50,
            'pcontext': 80,
            'ade20k': 160,
            'citys': 120,
            'sbu': 160,
        }
        args.epochs = epoches[args.dataset.lower()]
    if args.lr is None:
        lrs = {
            'coco': 0.004,
            'pascal_aug': 0.001,
            'pascal_voc': 0.0001,
            'pcontext': 0.001,
            'ade20k': 0.01,
            'citys': 0.01,
            'sbu': 0.001,
        }
        args.lr = lrs[args.dataset.lower()] / 8 * args.batch_size
    return args


def validation(model, val_dataloader, nClass, device='cpu'):
    """
    validation
    :param model:
    :param val_dataloader:
    :param nClass:
    :param device:
    :return:
    """
    total_inter = torch.zeros(nClass)
    total_union = torch.zeros(nClass)
    total_correct = 0
    total_label = 0

    model.eval()

    for i, (image, mask, mask_label) in tqdm(enumerate(val_dataloader), total=val_dataloader.__len__(),
                                             desc='Validating...'):
        image, mask, mask_label = image.to(device), mask.to(device), mask_label.to(device)

        with torch.no_grad():
            outputs = model(image)

            correct, labeled = batch_pix_accuracy(outputs, mask)
            inter, union = batch_intersection_union(outputs, mask, nClass)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            # 计算 pa、miou
            pixAcc = 1.0 * total_correct / (2.220446049250313e-16 + total_label)  # remove np.spacing(1)
            IoU = 1.0 * total_inter / (2.220446049250313e-16 + total_union)
            mIoU = IoU.mean().item()
            # print("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))
            # logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))

    return pixAcc, mIoU


def main():
    # 加载数据集
    print('Loading dataset...')
    train_dataset = VOCSegDataset(config.DATA_PATH, train=True, transform_sync=True, transform_img=True)
    val_dataset = VOCSegDataset(config.DATA_PATH, train=False, transform_sync=True, transform_img=True)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=config.BATCH_SIZE, num_workers=0, pin_memory=True)

    # define the model
    # model_fcn8x = FCN(num_class=len(config.VOC_CLASSES), in_channel=3, model_type='fcn8x').to(config.DEVICE)
    model = ResNetFCN(num_class=len(config.VOC_CLASSES)).to(config.DEVICE)

    # 定义优化器
    if config.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.LR,
            betas=config.BETAS
        )
    elif config.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY,
            momentum=config.MOMENTUM
        )

    # lr scheduling
    if config.SCHEDULER == 'mslr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.LR_STEPS,
                                                            gamma=config.LR_GAMMA_MSLR)
    elif config.SCHEDULER == 'slr':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.LR_STEP_SIZE,
                                                       gamma=config.LR_GAMMA_SLR)

    # 损失函数
    ce_loss_fun = CrossEntropy2d(nclass=len(config.VOC_CLASSES))
    dice_loss_fun = DiceLoss2d(nclass=len(config.VOC_CLASSES))

    # 加载预训练模型
    if config.INIT_EPOCH > 0:
        load_checkpoint(config.LATEST_MODEL, model, optimizer, lr_scheduler, lr=config.LR, device=config.DEVICE)

    # 最好模型效果标识
    best_pred = 0.0

    # training
    # logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))
    print('Start training from Epoch {}, Total Epochs: {}'.format(config.INIT_EPOCH, config.NUM_EPOCHS))

    model.train()
    for epoch in range(config.INIT_EPOCH, config.NUM_EPOCHS):
        total_loss = torch.zeros([]).to(config.DEVICE)
        total_ce_loss = torch.zeros([]).to(config.DEVICE)
        total_dice_loss = torch.zeros([]).to(config.DEVICE)

        for i, (images, masks, mask_labels) in tqdm(enumerate(train_dataloader), total=train_dataloader.__len__(),
                                                    desc='Epoch {} ...'.format(epoch)):
            images, masks, mask_labels = images.to(config.DEVICE), masks.to(config.DEVICE), mask_labels.to(
                config.DEVICE)

            outputs = model(images)

            ce_loss = ce_loss_fun.forward(outputs, masks)
            dice_loss = dice_loss_fun.forward(outputs, mask_labels)
            loss = ce_loss + dice_loss

            optimizer.zero_grad()  # loss.backward 之前将梯度清空
            loss.backward()
            optimizer.step()

            total_loss += loss
            total_ce_loss += ce_loss
            total_dice_loss += dice_loss

        total_loss = total_loss / train_dataloader.__len__()
        total_ce_loss = total_ce_loss / train_dataloader.__len__()
        total_dice_loss = total_dice_loss / train_dataloader.__len__()
        print('Epoch {}/{}, ce loss: {}, dice loss: {}, total loss: {}'.format(epoch, config.NUM_EPOCHS, total_ce_loss, total_dice_loss, total_loss))

        lr_scheduler.step()

        # save latest model
        save_mode_latest = os.path.join('./checkpoints', 'resnetfcn', config.LATEST_MODEL)
        save_checkpoint(model, optimizer, lr_scheduler, save_mode_latest)

        # validation
        if epoch % config.EVAL_EPOCH == 0:
            # do evaluation
            pixAcc, mIoU = validation(model, val_dataloader, len(config.VOC_CLASSES), config.DEVICE)
            print('Validation in Epoch {}, PA: {}, MIOU: {}'.format(epoch, pixAcc, mIoU))

            new_pred = (pixAcc + mIoU) / 2
            if new_pred > best_pred:
                best_pred = new_pred
                save_model_best = os.path.join('./checkpoints', 'resnetfcn', config.BEST_MODEL)
                save_checkpoint(model, optimizer, lr_scheduler, save_model_best)

            # 记得把模式改会 train
            model.train()


if __name__ == '__main__':
    # args = parse_args()
    main()
