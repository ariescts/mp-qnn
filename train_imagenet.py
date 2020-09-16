import os
import time, logging, argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from ast import literal_eval

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import models
from utils import *
from preprocess import get_transform
from data import create_dataloader_imagenet, create_dataloader_damagenet

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

MODEL_NAMES = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('_') and callable(models.__dict__[name]))


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')
    parser.add_argument('--arch', '-a', type=str, default='qresnet18_3', choices=MODEL_NAMES, help='model name')
    parser.add_argument('--dataset', '-d', type=str, default='imagenet', help='dataset name')
    parser.add_argument('--batch-size', '-b', type=int, default=512, help='dataloader batch-size')
    parser.add_argument('--workers', '-w', type=int, default=16, help='dataloader workers')
    parser.add_argument('--lr', type=float, default=1e-1, help='initial learning rate')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--epochs', type=int, default=40, help='total training epochs')
    parser.add_argument('--start-epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--print-freq', type=int, default=100, help='output print frequency (batch-size)')
    parser.add_argument('--resume', '-r', type=str, default='', help='path of previous checkpoint')
    parser.add_argument('--result-path', type=str, default='./results/train_tmp/', help='results main path')
    parser.add_argument('--save', type=str, default='2', help='save folder')
    args = parser.parse_args()
    return args


def train(model, dataloder, criterion, epoch, optimizer):
    model.train()
    loss, prec1, prec5 = forward(model, dataloder, criterion, epoch, training=True, optimizer=optimizer)
    return loss, prec1, prec5


def validate(model, dataloder, criterion, epoch):
    model.eval()
    loss, prec1, prec5 = forward(model, dataloder, criterion, epoch, training=False)
    return loss, prec1, prec5


def forward(model, dataloder, criterion, epoch=0, training=True, optimizer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (inputs, targets) in enumerate(dataloder):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        if type(outputs) is list:
            outputs = outputs[0]
        prec1, prec5 = accuracy(outputs, targets, (1, 5))

        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}] \t'
                         f'Time {batch_time.val:.4f} ({batch_time.avg:.4f}) \t'
                         f'Data {data_time.val:.4f} ({data_time.avg:.4f}) \t'
                         f'Loss {losses.val:.4f} ({losses.avg:.4f}) \t'
                         f'Prec@1 {top1.val:.4f} ({top1.avg:.4f}) \t'
                         f'Prec@5 {top5.val:.4f} ({top5.avg: .4f})'.format(
                epoch, i, len(dataloder), phase='TRAINING' if training else 'EVALUATING',
                batch_time=batch_time, data_time=data_time, losses=losses, top1=top1, top5=top5
            ))

    return losses.avg, top1.avg, top5.avg


def main(args):
    args.result_path = os.path.join(args.result_path, args.arch)
    if args.save == '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.result_path, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        raise ValueError('save folder already exist!')

    set_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    logging.info("creating dataloader %s", args.dataset)
    transform = {'train': get_transform(args.dataset, augment=True),
                 'eval': get_transform(args.dataset, augment=False)}
    train_loader, eval_loader = create_dataloader_imagenet(args.dataset, transform, args.batch_size, args.workers)
    transform_adv = get_transform('damagenet')
    adv_loader = create_dataloader_damagenet('damagenet', transform_adv, args.batch_size//2, args.workers)
    logging.info("created dataloader with batch-size: %d", args.batch_size)

    logging.info("creating model %s", args.arch)
    model = models.__dict__[args.arch]()

    cudnn.benchmark = True
    model = nn.DataParallel(model.cuda())

    logging.info("created model")
    logging.debug(model)

    if args.resume:
        logging.info('loading checkpoint %s' % args.resume)
        if not os.path.exists(args.resume):
            raise ValueError('The resume checkpoint does not exist!')
        checkpoint = args.resume
        if os.path.isdir(args.resume):
            results.load(os.path.join(args.resume, 'results.csv'))
            checkpoint = os.path.join(args.resume, 'checkpoint.pth.tar')
        checkpoint = torch.load(checkpoint)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45],
    #                                                  gamma=0.1, last_epoch=args.start_epoch-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs * 1.05),
                                                           last_epoch=args.start_epoch - 1)

    regime = {'optimizer': optimizer, 'lr': args.lr, 'wd': args.wd, 'scheduler': scheduler}
    logging.info('training regime: %s', regime)

    logging.info('start training process')

    train_best_prec1 = 0
    train_best_epoch = 0
    val_best_prec1 = 0
    val_best_epoch = 0
    adv_best_prec1 = 0
    adv_best_epoch = 0

    train_loss_plt = []
    train_prec1_plt = []
    val_loss_plt = []
    val_prec1_plt = []
    adv_loss_plt = []
    adv_prec1_plt = []

    for epoch in range(args.start_epoch, args.epochs):
        for param_group in optimizer.param_groups:
            logging.info('learning rate: %f' % param_group['lr'])

        train_loss, train_prec1, train_prec5 = train(model, train_loader, criterion, epoch, optimizer)
        train_loss_plt.append(train_loss)
        train_prec1_plt.append(train_prec1)
        scheduler.step()

        val_loss, val_prec1, val_prec5 = validate(model, eval_loader, criterion, epoch)
        val_loss_plt.append(val_loss)
        val_prec1_plt.append(val_prec1)

        adv_loss, adv_prec1, adv_prec5 = validate(model, adv_loader, criterion, epoch)
        adv_loss_plt.append(adv_loss)
        adv_prec1_plt.append(adv_prec1)

        flag_best_val = val_prec1 > val_best_prec1
        if flag_best_val:
            val_best_epoch = epoch
        val_best_prec1 = max(val_best_prec1, val_prec1)

        train_best_prec1 = max(train_best_prec1, train_prec1)
        adv_best_prec1 = max(adv_best_prec1, adv_prec1)

        checkpoint_train = {'epoch': epoch, 'arch': args.arch, 'val_best_prec1': val_best_prec1,
                            'val_best_epoch': val_best_epoch, 'regime': regime, 'state_dict': model.state_dict()}
        save_checkpoint(checkpoint_train, flag_best_val, save_path)

        logging.info(f'\n Epoch: {0} \t Training Loss {train_loss:.4f} \t Training Prec1 {train_prec1:.4f} \t'
                     f'Training Prec5 {train_prec5:.4f} \t Validation Loss {val_loss:.4f} \t '
                     f'Validation Prec1 {val_prec1:.4f} \t Validation Prec5 {val_prec5:.4f} \t '
                     f'Adversarial Loss {adv_loss:.4f} \t Adversarial Prec1 {adv_prec1:.4f} \t '
                     f'Adversarial Prec5 {adv_prec5:.4f} \n'.format(epoch, train_loss=train_loss, train_prec1=train_prec1,
                                                                    train_prec5=train_prec5, val_loss=val_loss,
                                                                    val_prec1=val_prec1, val_prec5=val_prec5,
                                                                    adv_loss=adv_loss, adv_prec1=adv_prec1,
                                                                    adv_prec5=adv_prec5))

        results.add(epoch=epoch, train_loss=train_loss, val_loss=val_loss, adv_loss=adv_loss,
                    train_prec1=train_prec1, val_pre1=val_prec1, adv_prec1=adv_prec1,
                    train_prec5=train_prec5, val_prec5=val_prec5, adv_prec5=adv_prec5,
                    val_best_epoch=val_best_epoch, val_best_prec1=val_best_prec1)
        results.save()

    return train_best_prec1, val_best_prec1, adv_best_prec1


if __name__ == '__main__':
    args = get_args()
    print(args)
    train_acc, val_acc, adv_acc = main(args)
    print('training acc %.4f \t validation acc %.4f \t adversarial acc %.4f' % (train_acc, val_acc, adv_acc))
