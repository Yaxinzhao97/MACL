import os
import sys
import time
import re
import argparse
import numpy as np
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from torchvision import datasets, models, transforms


plt.switch_backend('agg')

sys.path.append('../')
sys.path.append('.')

import utils.augmentation as A
from dataset.audio2vector_dataset_vox import audio_vector_dataset_vox_multi
from model import MemDPC_BD

from utils.utils import AverageMeter, save_checkpoint, Logger, \
    calc_topk_accuracy, neq_load_customized, MultiStepLR_Restart_Multiplier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='resnet18', type=str)
    parser.add_argument('--model', default='memdpc', type=str)
    parser.add_argument('--dataset', default='vox', type=str)
    parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
    parser.add_argument('--mem_size', default=512, type=int, help='memory size')
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--resume', default='', type=str, help='path of model to resume')
    parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
    parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
    parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
    parser.add_argument('--img_dim', default=256, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('-j', '--workers', default=8, type=int)
    args = parser.parse_args()
    return args

class VariablesChangeException(Exception):
    pass

best_acc = 0

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_gpu = len(str(args.gpu).split(','))
    args.batch_size = num_gpu * args.batch_size

    ### model ###
    if args.model == 'memdpc':
        model = MemDPC_BD(is_bidirectional=True,
                          dropout=0.5,
                          network=args.net,
                          mem_size=args.mem_size)
    else:
        raise NotImplementedError('wrong model!')

    model.to(device)
    model = nn.DataParallel(model)
    model_without_dp = model.module

    ### optimizer ###
    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()
    
    ### data ###
    transform = transforms.Compose([
        A.RandomGray(consistent=False, p=0.25),
        A.ColorJitter(0.5, 0.5, 0.5, 0.25, consistent=False, p=1.0),
        A.ToTensor(),
        A.Normalize()
     ])
    train_loader = get_data(transform, 'train')
    val_loader = get_data(transform, 'test')

    lr_milestones_eps = [5, 10]
    lr_milestones = [len(train_loader) * m for m in lr_milestones_eps]
    print('=> Use lr_scheduler: %s eps == %s iters' % (str(lr_milestones_eps), str(lr_milestones)))
    lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=lr_milestones, repeat=1)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    args.iteration = 1

    ### restart training ###
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resumed checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(device))
            args.start_epoch = 0
            args.iteration = checkpoint['iteration']
            best_acc = checkpoint['best_acc']
            model_without_dp.load_state_dict(checkpoint['state_dict'])
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print('[WARNING] Not loading optimizer states')
            print("=> loaded resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("[Warning] no checkpoint found at '{}'".format(args.resume))
            sys.exit(0)

    # logging tools
    args.img_path, args.model_path = set_path(args)
    args.logger = Logger(path=args.img_path)
    args.logger.log('args=\n\t\t' + '\n\t\t'.join(['%s:%s' % (str(k), str(v)) for k, v in vars(args).items()]))

    args.writer_val = SummaryWriter(logdir=os.path.join(args.img_path, 'val'))
    args.writer_train = SummaryWriter(logdir=os.path.join(args.img_path, 'train'))
    args.writer_iter = SummaryWriter(logdir=os.path.join(args.img_path, 'iter'))

    torch.backends.cudnn.benchmark = True

    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)

        train_loss, train_acc = train_one_epoch(train_loader,
                                                val_loader,
                                                model,
                                                criterion,
                                                optimizer,
                                                lr_scheduler,
                                                device,
                                                epoch,
                                                args,
                                               model_without_dp)
        
    print('Training from ep %d to ep %d finished'
          % (args.start_epoch, args.epochs))
    sys.exit(0)


def train_one_epoch(train_data_loader, val_data_loader, model, criterion, optimizer, lr_scheduler, device, epoch, args, model_without_dp):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = [[AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()],
                ]

    model.train()
    end = time.time()
    tic = time.time()
    iter_len = len(train_data_loader)
    global best_acc
    for idx, (audio_in, frames) in enumerate(train_data_loader):
        data_time.update(time.time() - end)
        audio_in = audio_in.to(device)
        frames = frames.to(device)
        B = audio_in.size(0)
        SEQ_LEN = audio_in.size(1)
        loss, loss_step, acc = model(audio_in, frames, False, args.iteration)

        for i in range(1):
            tops1, tops5, topi1, topi5 = acc[i].mean(0)  # average acc across multi-gpus
            accuracy[i][0].update(tops1.item(), B)
            accuracy[i][1].update(tops5.item(), B)
            accuracy[i][2].update(topi1.item(), B)
            accuracy[i][3].update(topi5.item(), B)

        loss = loss.mean()
        losses.update(loss.item(), B)
        params = [ np for np in model.named_parameters() if np[1].requires_grad ]
        initial_params = [ (name, p.clone()) for (name, p) in params ]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        vars_change = True

        batch_time.update(time.time() - end)
        end = time.time()

        if (iter_len * epoch + idx) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f}\t'
                  'Accs0: {acc[0][0].val:.4f}\t'
                  'Accs5: {acc[0][1].val:.4f}\t'
                  'Acci0: {acc[0][2].val:.4f}\t'
                  'Acci5: {acc[0][3].val:.4f}\t'
                  'T-data:{dt.val:.2f} T-batch:{bt.val:.2f}\t'.format(
                epoch, iter_len * epoch + idx, len(train_data_loader) * (epoch + 1),
                loss=losses, acc=accuracy, dt=data_time, bt=batch_time))

            args.writer_train.add_scalar('local/loss', losses.val, args.iteration)
            args.writer_train.add_scalar('local/F-tops1', accuracy[0][0].val, args.iteration)
            args.writer_train.add_scalar('local/F-tops5', accuracy[0][1].val, args.iteration)
            args.writer_train.add_scalar('local/F-topi1', accuracy[0][2].val, args.iteration)
            args.writer_train.add_scalar('local/F-topi5', accuracy[0][3].val, args.iteration)
        
        
        if args.iteration % 10000 == 0:
            print("=====================================val start=============================")
            val_loss, val_acc = validate(val_data_loader,
                                     model,
                                     criterion,
                                     device,
                                     args.iteration,
                                     args)
            
            print("=====================================val end=============================")

            # save check_point
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            save_dict = {'epoch': args.iteration,
                         'state_dict': model_without_dp.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'iteration': args.iteration}
            save_checkpoint(save_dict, is_best,
                            filename=os.path.join(args.model_path, 'iter%s.pth.tar' % str(args.iteration)),
                            keep_all=False)

            args.writer_iter.add_scalar('iter/loss', losses.avg, args.iteration)
            args.writer_iter.add_scalar('iter/F-tops1', accuracy[0][0].avg, args.iteration)
            args.writer_iter.add_scalar('iter/F-tops5', accuracy[0][1].avg, args.iteration)
            args.writer_iter.add_scalar('iter/F-topi1', accuracy[0][2].avg, args.iteration)
            args.writer_iter.add_scalar('iter/F-topi5', accuracy[0][3].avg, args.iteration)
            

        args.iteration += 1
        if lr_scheduler is not None: lr_scheduler.step()

    print('Epoch: [{0}]\t'
          'T-epoch:{t:.2f}\t'.format(epoch, t=time.time() - tic))

    args.writer_train.add_scalar('global/loss', losses.avg, epoch)
    args.writer_train.add_scalar('global/F-tops1', accuracy[0][0].avg, epoch)
    args.writer_train.add_scalar('global/F-tops5', accuracy[0][1].avg, epoch)
    args.writer_train.add_scalar('global/F-topi1', accuracy[0][2].avg, epoch)
    args.writer_train.add_scalar('global/F-topi5', accuracy[0][3].avg, epoch)

    return losses.avg, accuracy[0][0].avg + accuracy[0][2].avg


def validate(data_loader, model, criterion, device, epoch, args):
    losses = AverageMeter()
    accuracy = [[AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()],
                ]

    model.eval()

    with torch.no_grad():
        for idx, (audio_in, frames) in enumerate(data_loader):
            audio_in = audio_in.to(device)
            frames = frames.to(device)

            B = audio_in.size(0)
            SEQ_LEN = audio_in.size(1)
            loss, loss_step, acc = model(audio_in, frames, add_noise=False)

            for i in range(1):
                tops1, tops5, topi1, topi5 = acc[i].mean(0)
                accuracy[i][0].update(tops1.item(), B)
                accuracy[i][1].update(tops5.item(), B)
                accuracy[i][2].update(topi1.item(), B)
                accuracy[i][3].update(topi5.item(), B)

            loss = loss.mean()
            losses.update(loss.item(), B)
            if idx % args.print_freq == 0:
                
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f}\t'
                  'Accs0: {acc[0][0].val:.4f}\t'
                  'Accs5: {acc[0][1].val:.4f}\t'
                  'Acci0: {acc[0][2].val:.4f}\t'
                  'Acci5: {acc[0][3].val:.4f}\t'.format(
                epoch, idx, len(data_loader),
                loss=losses, acc=accuracy))

    print('Epoch: [{0}/{1}]\t'
          'Loss {loss.avg:.6f}\t'
          'Accs0: {acc[0][0].avg:.4f}\t'
          'Accs5: {acc[0][1].avg:.4f}\t'
          'Acci0: {acc[0][2].avg:.4f}\t'
          'Acci5: {acc[0][3].avg:.4f}\t'.format(
        epoch, args.epochs,
        loss=losses, acc=accuracy))

    args.writer_val.add_scalar('global/loss', losses.avg, epoch)
    args.writer_val.add_scalar('global/F-tops1', accuracy[0][0].avg, epoch)
    args.writer_val.add_scalar('global/F-tops5', accuracy[0][1].avg, epoch)
    args.writer_val.add_scalar('global/F-topi1', accuracy[0][2].avg, epoch)
    args.writer_val.add_scalar('global/F-topi5', accuracy[0][3].avg, epoch)
    
    model.train()

    return losses.avg, accuracy[0][0].avg + accuracy[0][2].avg


def get_data(transform, mode='train'):
    

    dataset = audio_vector_dataset_vox_multi(mode, transform=transform)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True if mode == 'train' else False,
                                  num_workers=args.workers,
                                  pin_memory=True,
                                  drop_last=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader


def set_path(args):
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = 'log_{args.prefix}/{args.model}-{args.img_dim}_{args.net}_\
mem{args.mem_size}_bs{args.batch_size}_lr{args.lr}'.format(args=args)

    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    return img_path, model_path


if __name__ == '__main__':
    args = parse_args()
    main(args)