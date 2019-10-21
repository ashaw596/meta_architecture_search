import argparse
import glob
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset

import utils
from our_models import *

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--model', type=str, default='get_cifar_tuned_model(True)', help='Model to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--load_path', type=str, default=None, help='')
parser.add_argument('--num_workers', type=int, default=2, help='args.num_workers')
parser.add_argument('--start_epoch', type=int, default=0, help='start_epoch')
parser.add_argument('--save_frequency', type=int, default=50)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--eval', type=int, default=0)
args = parser.parse_args()



def main():
    if args.load_path:
        args.save = Path(args.load_path) / 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    else:
        args.save = Path('logs') / 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)

    model = eval(args.model)
    if args.gpu:
        model = model.cuda()

    if args.load_path:
        utils.load(model, os.path.join(args.load_path, 'weights.pt'))
        print("loaded")

    direct_model = model
    if args.gpu:
        model = torch.nn.DataParallel(model)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    if args.eval:
        direct_model.drop_path_prob = 0
        valid_acc, valid_obj = infer(valid_queue, model, args.gpu)
        logging.info('valid_acc %f', valid_acc)
        return

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step(epoch)
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        direct_model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, optimizer, args.gpu)
        logging.info('train_acc %f', train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, args.gpu)
        logging.info('valid_acc %f', valid_acc)

        if epoch >= args.epochs - 50 or epoch % args.save_frequency == 0:
            utils.save(model.module, os.path.join(args.save, f'weights_{epoch}.pt'))


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
    for step, (input, target) in enumerate(valid_queue):
        with torch.no_grad():
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

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
