"""
training
"""
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import config
from fcn import FCN
import os
from tqdm import tqdm
from dataset import VOCSegDataset
from fcn import FCN
from utils import load_checkpoint, save_checkpoint, batch_intersection_union, batch_pix_accuracy, CrossEntropy2d
from networks.hrnet import HRnet


# torch.backends.cudnn.benchmark = False


def validation(model, val_dataloader, nClass, device):
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

    for i, (image, mask, mask_label) in tqdm(enumerate(val_dataloader), total=val_dataloader.__len__(), desc='Validating...'):

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
    train_dataset = VOCSegDataset(config.DATA_PATH, transform_struct=config.transforms_struct, transform_data=config.transforms_data)
    val_dataset = VOCSegDataset(config.DATA_PATH, transform_struct=config.transforms_struct, transform_data=config.transforms_data, train=False)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=config.BATCH_SIZE, num_workers=0, pin_memory=True)

    # model
    # model_fcn8x = FCN(num_class=len(config.VOC_CLASSES), in_channel=3, model_type='fcn8x').to(config.DEVICE)
    model = HRnet(num_classes=len(config.VOC_CLASSES), backbone='hrnetv2_w18', pretrained=False).to(config.DEVICE)

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
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.LR_STEPS, gamma=config.LR_GAMMA_MSLR)
    elif config.SCHEDULER == 'slr':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA_SLR)

    # 损失函数
    ce_loss_fun = CrossEntropy2d(nclass=len(config.VOC_CLASSES), ignore_label=255)

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

        for i, (images, masks, mask_labels) in tqdm(enumerate(train_dataloader), total=train_dataloader.__len__(), desc='Epoch {} ...'.format(epoch)):
            images, masks, mask_labels = images.to(config.DEVICE), masks.to(config.DEVICE), mask_labels.to(config.DEVICE)

            outputs = model(images)

            ce_loss = ce_loss_fun.forward(outputs, masks)
            dice_loss = torch.zeros_like(ce_loss)
            loss = ce_loss + dice_loss

            optimizer.zero_grad()   # loss.backward 之前将梯度清空
            loss.backward()
            optimizer.step()

            total_loss += loss
            total_ce_loss += ce_loss
            total_dice_loss += dice_loss

        total_loss = total_loss / train_dataloader.__len__()
        total_ce_loss = total_ce_loss / train_dataloader.__len__()
        total_dice_loss = total_dice_loss / train_dataloader.__len__()
        print('Epoch {}, ce loss: {}, dice loss: {}, total loss: {}'.format(epoch, total_ce_loss, total_dice_loss, total_loss))

        lr_scheduler.step()

        # save latest model
        net = 'hrnet'
        model_name = os.path.join('./checkpoints/', net, config.LATEST_MODEL)
        save_checkpoint(model, optimizer, lr_scheduler, model_name)

        # validation
        if epoch % config.EVAL_EPOCH == 0:
            # do evaluation
            pixAcc, mIoU = validation(model, val_dataloader, len(config.VOC_CLASSES), config.DEVICE)
            print('Validation in Epoch {}, PA: {}, MIOU: {}'.format(epoch, pixAcc, mIoU))

            new_pred = (pixAcc + mIoU) / 2
            if new_pred > best_pred:
                best_pred = new_pred
                model_name = os.path.join('./checkpoints/', net, config.BEST_MODEL)
                save_checkpoint(model, optimizer, lr_scheduler, model_name)

            # 记得把模式改会 train
            model.train()


if __name__ == '__main__':
    main()
