import argparse
import bisect
import glob
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
# from thop_folder import profile
from torch.optim.lr_scheduler import CosineAnnealingLR

import utils
from our_models import *

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='data/imagenet/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.05, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--epochs', type=int, default=375, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--model', type=str, default='get_imagenet_tuned_model(True)', help='Model to use')
# parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=10., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
parser.add_argument('--load_checkpoint', type=str, default=None)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--start_epoch', type=int, default=-1)
parser.add_argument('--period', type=str, default='25,50,100,200')
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--eval', type=int, default=0)
args = parser.parse_args()

args.save = 'eval-imagenet-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 1000




def main():
    if args.load_checkpoint:
        args.save = Path(args.load_checkpoint) / 'eval-imagenet-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    else:
        args.save = Path('logs') / 'eval-imagenet-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(args.save / 'log.txt')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    model = eval(args.model)

    # flops, params = profile(model, input_size=(1, 3, 224, 224))
    # print("flops" + str(flops) + " params" + str(params))
    if args.load_checkpoint:
        dictionary = torch.load(args.load_checkpoint)
        start_epoch = dictionary['epoch'] if args.start_epoch == -1 else args.start_epoch
        model.load_state_dict(dictionary['state_dict'])
    else:
        start_epoch = 0 if args.start_epoch == -1 else args.start_epoch

    direct_model = model

    if args.gpu:
        model = nn.DataParallel(model)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # if args.load_checkpoint:
    #   optimizer.load_state_dict(dictionary['optimizer'])
    #   del dictionary

    traindir = os.path.join(args.data, 'train')
    validdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = dset.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    if args.eval:
        direct_model.drop_path_prob = 0
        valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, args.gpu)
        logging.info('valid_acc_top1 %f', valid_acc_top1)
        logging.info('valid_acc_top5 %f', valid_acc_top5)
        return

    if args.period is not None:
        periods = args.period.split(',')
        periods = [int(p) for p in periods]
        totals = []
        total = 0
        for p in periods:
            total += p
            totals.append(total)
        scheduler = CosineAnnealingLR(optimizer, periods[0])
    else:
        periods = None
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)

    best_acc_top1 = 0
    for epoch in range(start_epoch, args.epochs):
        if args.period is None:
            scheduler.step(epoch)
        else:
            assert len(periods) > 0
            index = bisect.bisect_left(totals, epoch)
            scheduler.T_max = periods[index]
            if index == 0:
                e = epoch
            else:
                e = epoch - totals[index - 1]
            scheduler.step(e % periods[index])
            logging.info("schedule epoch:" + str(e % periods[index]))
            logging.info("schedule period:" + str(periods[index]))
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        direct_model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, optimizer, args.gpu)
        logging.info('train_acc %f', train_acc)

        valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, args.gpu)
        logging.info('valid_acc_top1 %f', valid_acc_top1)
        logging.info('valid_acc_top5 %f', valid_acc_top5)

        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True

        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'best_acc_top1': best_acc_top1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save)


def train(train_queue, model, optimizer, gpu):
    objs = utils.AverageTracker()
    top1 = utils.AverageTracker()
    top5 = utils.AverageTracker()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        # input = input.cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)
        if gpu:
            target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        dic = model(input, target)
        logits = dic['logits']
        loss = dic['loss']
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        del loss, logits, prec1, prec5, input, target

    return top1.avg, objs.avg


def infer(valid_queue, model, gpu):
    objs = utils.AverageTracker()
    top1 = utils.AverageTracker()
    top5 = utils.AverageTracker()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            # input = input.cuda(non_blocking=True)
            if gpu:
                target = target.cuda(non_blocking=True)

            dic = model(input, target)
            logits = dic['logits']
            loss = dic['loss']

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            assert isinstance(n, int)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            del loss, logits, prec1, prec5, input, target

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main()
