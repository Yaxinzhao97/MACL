import os
import sys
import time
import argparse
import re
import numpy as np
import random
import json
from tqdm import tqdm
from tensorboardX import SummaryWriter
import math
from scipy.signal import savgol_filter

sys.path.append('../')
sys.path.append('../memdpc/')
sys.path.append('.')
from dataset.audio2vector_dataset_vox import audio_vector_dataset_self_register_multi as audio_vector_dataset_self_register
from model import ModelLmk
from utils.utils import AverageMeter, ConfusionMeter, save_checkpoint, \
calc_topk_accuracy, denorm, calc_accuracy, neq_load_customized, Logger

import torch
import torch.optim as optim
from torch.utils import data
import torch.nn as nn
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
from common.icp import icp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='resnet18', type=str)
    parser.add_argument('--dataset', default='vox', type=str)
    parser.add_argument('--ds', default=3, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--schedule', default=[], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--resume', default='', type=str)

    parser.add_argument('--pretrain', default=' ', type=str)
    parser.add_argument('--test', default='', type=str)
    parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--print_freq', default=5, type=int)
    parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
    parser.add_argument('--train_what', default='ft', type=str, help='Train what parameters?')
    parser.add_argument('--prefix', default='tmp', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('--mdn_max', action='store_true', help="for inference")
    parser.add_argument('--mdn_sum', action='store_true', help="for inference")
    parser.add_argument('--random_clip_num', default=2, type=int)
    parser.add_argument('--jpg_freq', default=50, type=int)
    parser.add_argument('--show_animation', default=True, action='store_true')
    parser.add_argument('-M', default=3, type=int)
    parser.add_argument('--sigmaElu', default=1, type=int)
    args = parser.parse_args()
    return args

class VariablesChangeException(Exception):
    pass

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_gpu = len(str(args.gpu).split(','))
    args.batch_size = num_gpu * args.batch_size

    model = ModelLmk(M=args.M,train_what=args.train_what)
    model.to(device)
    model = nn.DataParallel(model)
    model_without_dp = model.module
    criterion = nn.L1Loss()

    ### optimizer ###
    params = None
    if args.train_what == 'ft':
        print('=> finetune backbone with smaller lr')
        params = []
        for name, param in model.module.named_parameters():
            if ('GRU' in name) or ('audioEncoder' in name):
                params.append({'params': param, 'lr': args.lr / 10})
            else:
                params.append({'params': param})
    elif args.train_what == 'last':
        print('=> train only last layer')
        params = []
        for name, param in model.named_parameters():
            if ('GRU' in name) or ('audioEncoder' in name):
                param.requires_grad = False
            else:
                params.append({'params': param})
    else:
        pass  # train all layers

    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')
    if params is None: params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    ### scheduler ###
    if args.dataset == 'mydata':
        step = args.schedule
        if step == []: step = [50, 80, 300]
        lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=step, repeat=1)
    elif args.dataset == 'vox':
        step = args.schedule
        if step == []: step = [300, 400]
        lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=step, repeat=1)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    print('=> Using scheduler at {} epochs'.format(step))

    args.old_lr = None
    best_acc = 10000
    args.iteration = 1

    ### if in test mode ###
    if args.test:
        if os.path.isfile(args.test):
            print("=> loading test checkpoint '{}'".format(args.test))
            checkpoint = torch.load(args.test, map_location=torch.device('cpu'))
            try:
                model_without_dp.load_state_dict(checkpoint['state_dict'])
            except:
                print('=> [Warning]: weight structure is not equal to test model; Load anyway ==')
                model_without_dp = neq_load_customized(model_without_dp, checkpoint['state_dict'])
            epoch = checkpoint['epoch']
            print("=> loaded testing checkpoint '{}' (epoch {})".format(args.test, checkpoint['epoch']))
        elif args.test == 'random':
            epoch = 0
            print("=> loaded random weights")
        else:
            print("=> no checkpoint found at '{}'".format(args.test))
            sys.exit(0)

        args.logger = Logger(path=os.path.dirname(args.test))
        _, test_dataset = get_data(None, 'test')
        test_loss, test_acc = test(test_dataset, model, criterion, device, epoch, args)
        sys.exit()

    ### restart training ###
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resumed checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            args.iteration = checkpoint['iteration']
            best_acc = checkpoint['best_acc']
            model_without_dp.load_state_dict(checkpoint['state_dict'])
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print('[WARNING] Not loading optimizer states')
            print("=> loaded resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            sys.exit(0)
    # TODO
    if (not args.resume) and args.pretrain:
        if args.pretrain == 'random':
            print('=> using random weights')
        elif os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            model_without_dp = neq_load_customized(model_without_dp, checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))
            sys.exit(0)

    ### data ###
    train_loader, _ = get_data(args, 'train')
    val_loader, _ = get_data(args, 'dev')

    # setup tools
    args.img_path, args.model_path, args.video_path, args.lmk_path = set_path(args)
    args.writer_val = SummaryWriter(logdir=os.path.join(args.img_path, 'val'))
    args.writer_train = SummaryWriter(logdir=os.path.join(args.img_path, 'train'))
    torch.backends.cudnn.benchmark = True

    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train_one_epoch(train_loader,
                                                model,
                                                criterion,
                                                optimizer,
                                                device,
                                                epoch,
                                                args)
        # TODO
        val_loss, val_acc = validate(val_loader,
                                     model,
                                     criterion,
                                     device,
                                     epoch,
                                     args)
        
        lr_scheduler.step(epoch)

        # save check_point
        is_best = val_acc < best_acc
        best_acc = min(val_acc, best_acc)
        save_dict = {
            'epoch': epoch,
            'backbone': args.net,
            'state_dict': model_without_dp.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'iteration': args.iteration}
        save_checkpoint(save_dict, is_best,
                        filename=os.path.join(args.model_path, 'epoch%s.pth.tar' % str(epoch)),
                        keep_all=False)

    print('Training from ep %d to ep %d finished'
          % (args.start_epoch, args.epochs))
    sys.exit(0)


def train_one_epoch(data_loader, model, criterion, optimizer, device, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    t_shape_idx = (27, 28, 29, 30, 33, 36, 39, 42, 45)
    anchor_t_shape = np.loadtxt(
                r'dataset/ANCHOR_T_SHAPE_{}.txt'.format(len(t_shape_idx)))
    s = np.abs(anchor_t_shape[5, 0] - anchor_t_shape[8, 0])
    anchor_t_shape = anchor_t_shape / s * 1.0
    c2 = np.mean(anchor_t_shape[[4,5,8], :], axis=0)
    anchor_t_shape -= c2

    if args.train_what == 'last':
        model.eval()
        model.module.final_bn.train()
        model.module.lmkNet.train()
        print('[Warning] train model with eval mode, except final and lmkNet layer')
    else:
        model.train()

    end = time.time()
    tic = time.time()
    dict_ct = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
    dict_rot = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
    dict_rot_freq = {0:0, 1:0, 2:0, 3:0, 5:0, 6:0, 7:0}
    dict_speaker = {0:{0:0, 1: 0, 2:0}, 1:{0:0, 1: 0, 2:0}, 2:{0:0, 1: 0, 2:0}}

    for idx, (audio_in, lmk_gt, audio, video_name, rot_trans, rot_quats, pid) in enumerate(data_loader):
        data_time.update(time.time() - end)
        audio_in = audio_in.to(device)
        lmk_gt = lmk_gt.to(device).float()
        rot_quats = rot_quats.to(device).float()
        rot_trans = rot_trans.to(device).float()
        B = audio_in.size(0)
        SEQ_LEN = audio_in.size(1)
        ids = close_face_lip(lmk_gt.detach().cpu().numpy())
        """register face"""
        input_face_ids = torch.cat([lmk_gt[i: i+1, idx, :] for i, idx in enumerate(ids)], dim=0)
        input_face_ids = input_face_ids.requires_grad_(False)
        
        if len(input_face_ids.shape) == 2:
            input_face_ids = input_face_ids.unsqueeze(1).repeat(1, audio_in.shape[1], 1)
        """
        outH:(B, SEQ_LEN, 136*M)
        sigma:(B, SEQ_LEN, 6)
        logits:(B, SEQ_LEN, 3)
        """
        outH, sigma, logits = model(audio_in, input_face_ids)
        
        pi = torch.clamp(nn.Softmax(-1)(logits), 1e-4, 1. - 1e-4)
        mdn_sigma = torch.clamp(nn.ELU()(sigma) + args.sigmaElu, 1e-4, 1e5)
        gt = torch.cat([lmk_gt, rot_quats.unsqueeze(-1), rot_trans[:, :, :, 2]], dim=-1)
        loss = mdn_loss_dense(gt, outH, mdn_sigma, pi)
        prior = torch.tensor([2.0 for _ in range(args.M)], requires_grad=False, dtype=torch.float)
        if torch.cuda.is_available():
            prior = prior.cuda()
        loss_prior = Dirichlet_loss(pi, args.M, prior)
        loss = loss + 0.005 * loss_prior
        
        mu_params = torch.cat([x.view(-1) for x in model.module.lmkNet.outheader.parameters()] )
        sigma_params = torch.cat([x.view(-1) for x in model.module.lmkNet.sigmaheader.parameters()] )
        pi_params = torch.cat([x.view(-1) for x in model.module.lmkNet.logitsheader.parameters()] )
        mu_l2_reg = torch.norm(mu_params, 2)
        sigma_l2_reg = torch.norm(sigma_params, 2)
        pi_l2_reg = torch.norm(pi_params, 2)

        
        loss = loss + 0.001 * pi_l2_reg + 0.01 * sigma_l2_reg
        losses.update(loss.item(), B * SEQ_LEN)
        params = [ np for np in model.named_parameters() if np[1].requires_grad ]
         # take a copy
        initial_params = [ (name, p.clone()) for (name, p) in params if 'bn' not in name]
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0)
        optimizer.step()
        vars_change = True

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.local_avg:.4f})\t'
                  'T-data:{dt.val:.2f} T-batch:{bt.val:.2f}\t'.format(
                epoch, idx, len(data_loader),
                loss=losses, dt=data_time, bt=batch_time))

            args.writer_train.add_scalar('local/loss', losses.val, args.iteration)

        args.iteration += 1
    print('Epoch: [{0}]\t'
          'Loss {loss.avg:.4f}\t'
          'T-epoch:{t:.2f}\t'.format(epoch, loss=losses, t=time.time() - tic))
    print(dict_ct)
    for key, _ in dict_rot_freq.items():
        if dict_ct[key] != 0:
            dict_rot_freq[key] = dict_rot[key] / dict_ct[key]
    print(dict_rot_freq)
    print(dict_speaker)
    args.writer_train.add_scalar('global/loss', losses.avg, epoch)

    return losses.avg

def validate(data_loader, model, criterion, device, epoch, args):
    losses = AverageMeter()
    t_shape_idx = (27, 28, 29, 30, 33, 36, 39, 42, 45)
    anchor_t_shape = np.loadtxt(
                r'dataset/ANCHOR_T_SHAPE_{}.txt'.format(len(t_shape_idx)))
    s = np.abs(anchor_t_shape[5, 0] - anchor_t_shape[8, 0])
    anchor_t_shape = anchor_t_shape / s * 1.0
    c2 = np.mean(anchor_t_shape[[4,5,8], :], axis=0)
    anchor_t_shape -= c2
    model.eval()
    res = []
    dict_ct = {0:0, 1:0, 2:0, 3:0, 4:0}
    dict_rot = {0:0, 1:0, 2:0, 3:0, 4:0}
    dict_rot_freq = {0:0, 1:0, 2:0, 3:0, 4:0}
    dict_speaker = {0:{0:0, 1: 0, 2:0}, 1:{0:0, 1: 0, 2:0}, 2:{0:0, 1: 0, 2:0}}
    with torch.no_grad():
        for idx, (audio_in, lmk_gt, audio, video_name, rot_trans, rot_quats, pid) in enumerate(data_loader):
            audio_in = audio_in.to(device)
            lmk_gt = lmk_gt.to(device).float()
            rot_quats = rot_quats.to(device).float()
            rot_trans = rot_trans.to(device).float()
            B = audio_in.size(0)
            SEQ_LEN = audio_in.size(1)
            ids = close_face_lip(lmk_gt.detach().cpu().numpy())
            """register face"""
            input_face_ids = torch.cat([lmk_gt[i: i+1, idx, :] for i, idx in enumerate(ids)], dim=0)
            input_face_ids = input_face_ids.requires_grad_(False)
            if len(input_face_ids.shape) == 2:
                input_face_ids = input_face_ids.unsqueeze(1).repeat(1, audio_in.shape[1], 1)
            """
            outH:(B, SEQ_LEN, 136*M)
            sigma:(B, SEQ_LEN, 6)
            logits:(B, SEQ_LEN, 3)
            """
            outH, sigma, logits = model(audio_in, input_face_ids)

            pi = torch.clamp(nn.Softmax(-1)(logits), 1e-4, 1. - 1e-4)
            mdn_sigma = torch.clamp(nn.ELU()(sigma) + args.sigmaElu, 1e-4, 1e5)
            gt = torch.cat([lmk_gt, rot_quats.unsqueeze(-1), rot_trans[:, :, :, 2]], dim=-1)

            loss = mdn_loss_dense(gt, outH, mdn_sigma, pi)
            
            mu_params = torch.cat([x.view(-1) for x in model.module.lmkNet.outheader.parameters()] )
            sigma_params = torch.cat([x.view(-1) for x in model.module.lmkNet.sigmaheader.parameters()] )
            pi_params = torch.cat([x.view(-1) for x in model.module.lmkNet.logitsheader.parameters()] )
            mu_l2_reg = torch.norm(mu_params, 2)
            sigma_l2_reg = torch.norm(sigma_params, 2)
            pi_l2_reg = torch.norm(pi_params, 2)

            losses.update(loss.item(), B * SEQ_LEN)
            if idx % args.jpg_freq == 0:
                video_display(pi, outH, mdn_sigma, lmk_gt, audio, len(data_loader), epoch, idx, video_name, args, pos_std=rot_trans)
                dis = compare_landmarks(pi, outH, mdn_sigma, lmk_gt, args, pos_std=rot_trans)
                res.append(dis)
            
    print('Loss {loss.avg:.4f}\t'.format(loss=losses))
    print('Distance {:.4f}\t'.format(sum(res) / len(res)))
    args.writer_val.add_scalar('global/loss', losses.avg, epoch)
    args.writer_val.add_scalar('global/dis', sum(res) / len(res), epoch)
    print(dict_ct)

    return losses.avg, sum(res) / len(res)


def test(dataset, model, criterion, device, epoch, args):
    prob_dict = {}
    model.eval()
    with torch.no_grad():
        end = time.time()
        data_sampler = data.RandomSampler(dataset)
        data_loader = data.DataLoader(dataset,
                                      batch_size=1,
                                      sampler=data_sampler,
                                      shuffle=False,
                                      num_workers=8,
                                      pin_memory=True)

        """
        audio_in:(B, SEQ_LEN, 24, 80)
        lmk_gt:(B, SEQ_LEN, 136)
        """
        for idx, (audio_in, lmk_gt) in tqdm(enumerate(data_loader), total=len(data_loader)):
            B = 1
            audio_in = audio_in.to(device)
            lmk_gt = lmk_gt.to(device)

            input_seq = input_seq.squeeze(0) # squeeze the '1' batch dim
            output, _ = model(input_seq)

            prob_mean = nn.functional.softmax(output, 2).mean(1).mean(0, keepdim=True)

            vname = vname[0]
            if vname not in prob_dict.keys():
                prob_dict[vname] = []
            prob_dict[vname].append(prob_mean)

        # show intermediate result
        if (title == 'ten') and (flip_idx == 0) and (aug_idx == 5):
            print('center-crop result:')
            acc_1 = summarize_probability(prob_dict,
                data_loader.dataset.encode_action, 'center')
            args.logger.log('center-crop:')
            args.logger.log('test Epoch: [{0}]\t'
                'Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
                .format(epoch, acc=acc_1))

    # show intermediate result
    if (title == 'ten') and (flip_idx == 0):
        print('five-crop result:')
        acc_5 = summarize_probability(prob_dict,
                data_loader.dataset.encode_action, 'five')
        args.logger.log('five-crop:')
        args.logger.log('test Epoch: [{0}]\t'
            'Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
            .format(epoch, acc=acc_5))

    # show final result
    print('%s-crop result:' % title)
    acc_final = summarize_probability(prob_dict,
        data_loader.dataset.encode_action, 'ten')
    args.logger.log('%s-crop:' % title)
    args.logger.log('test Epoch: [{0}]\t'
                    'Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
                    .format(epoch, acc=acc_final))
    sys.exit(0)

def get_data(args, mode='train'):
    print('Loading data for "%s" ...' % mode)
    global dataset
    if args.dataset == 'mydata':
#         dataset = audio_vector_dataset_self(model=mode)
        dataset = audio_vector_dataset_self_register(model=mode)
    elif args.dataset == 'vox':
        dataset = audio_vector_dataset_self_register(model=mode)
        # print("待实现。。。。。。")
    else:
        raise ValueError('dataset not supported')
    dataset.__getitem__(20)
    my_sampler = data.RandomSampler(dataset)
    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=args.workers,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'dev':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
#                                       sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=args.workers,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'test':
        data_loader = data.DataLoader(dataset,
                                      batch_size=1,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=args.workers,
                                      pin_memory=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader, dataset

def log_sum_exp(x, axis,mdn_max):
    ''' Source: https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation'''
    """Log-sum-exp trick implementation"""
    x_max = torch.max(x,axis,keepdim=True,out=None)[0]
    if mdn_max:
        return x_max
    return torch.log(torch.sum(torch.exp(x - x_max),
                       axis, keepdim=True))+x_max

"""
gt:(B, SEQ_LEN, 136)
mu:(B, SEQ_LEN, 136*M)
sigma:(B, SEQ_LEN, M*2)
pi:(B, SEQ_LEN, M)
"""
def mdn_loss_dense(gt, mu, sigma, pi):
    C = 68
    BS, SEQ_LEN, _ = gt.shape
    M = pi.shape[-1]
    
    w = torch.abs(gt[:, :, 66 * 2 + 1] - gt[:, :, 62 * 2 + 1])
    w = torch.tensor([5.0]).to('cuda') / (w * 4.0 + 0.1)
    w = w.unsqueeze(2)
    lip_region_w = torch.ones((mu.shape[0], mu.shape[1], M, 139)).to('cuda')
    lip_region_w[:, :, :, 48 * 2: 68 * 2] = torch.cat([w] * 40, dim=2).unsqueeze(2).repeat(1, 1, M, 1)
    V = lip_region_w.detach().clone().requires_grad_(False)
    
    gt = gt.unsqueeze(2)
    gt = gt.repeat(1, 1, M, 1)

    mu = mu.reshape(BS, SEQ_LEN, M, 2 * C + 1 + 2)
    sigma_tmp = sigma.view(BS, SEQ_LEN, M, 1)
    sigma_tmp = sigma_tmp[:, :, :, np.concatenate([np.arange(1) for _ in np.arange(C * 2 + 1 + 2)])]
    
    e = .5 * ((gt - mu) * torch.reciprocal(sigma_tmp)) ** 2
    e = torch.sum(e, -1)
    
    PI = torch.tensor(np.pi)
    if torch.cuda.is_available():
        PI = PI.cuda()
    coef = -torch.log(sigma) - torch.log(2 * PI)
    exponent = torch.log(pi) + coef -e
    
    loss = -torch.squeeze(log_sum_exp(exponent, 2, False), 2)
    loss = torch.sum(loss)

    loss = loss / (BS * SEQ_LEN)
    return loss

def Dirichlet_loss(alpha, m, prior):
    '''
    add dirichlet conjucate prior to the loss function to prevent all data fitting into single kernel
    '''

    C = 68
    loss = torch.sum((prior-1.0) * torch.log(alpha), axis=2)
    res = -torch.mean(loss)
    return res

def lmk_loss(pi, outH, mdn_sigma, lmk_gt, args):
    pi_max, max_pi_ind = torch.max(pi, -1)
    M = args.M
    C = outH.shape[-1] // M
    mdn_mu = outH.view(outH.shape[0], outH.shape[1], M, 136)
    mdn_sigma = mdn_sigma.view(outH.shape[0], outH.shape[1], M, 2)

    if args.mdn_max:
        a = max_pi_ind.unsqueeze(-1).repeat(1, 1, C).unsqueeze(2)
        lmk_pred = torch.gather(mdn_mu, 2, a)
        a = max_pi_ind.unsqueeze(-1).repeat(1, 1, 2).unsqueeze(2)
        sigmas = torch.gather(mdn_sigma, 2, a)
    elif args.mdn_sum:
        lmk_pred = torch.sum(mdn_mu * pi.unsqueeze(-1), 2)
        sigmas = torch.sum(mdn_sigma * pi.unsqueeze(-1), 2)
    lmk_pred = lmk_pred.squeeze(2)
    ''' lip region weight '''
    w = torch.abs(lmk_gt[:, :, 66 * 2 + 1] - lmk_gt[:, :, 62 * 2 + 1])
    w = torch.tensor([1.0]).to('cuda') / (w * 4.0 + 0.1)
    w = w.unsqueeze(2)
    lip_region_w = torch.ones((lmk_pred.shape[0], lmk_pred.shape[1], 136)).to('cuda')
    lip_region_w[:, :, 48 * 2:] = torch.cat([w] * 40, dim=2)
    lip_region_w = lip_region_w.detach().clone().requires_grad_(False)
    loss = torch.mean(torch.abs(lmk_pred - lmk_gt) * lip_region_w)
    return loss

def laplacian_loss(pi, outH, mdn_sigma, lmk_gt, args):
    pi_max, max_pi_ind = torch.max(pi, -1)
    M = args.M
    C = outH.shape[-1] // M

    mdn_mu = outH.view(outH.shape[0], outH.shape[1], M, 136)
    mdn_sigma = mdn_sigma.view(outH.shape[0], outH.shape[1], M, 2)

    if args.mdn_max:
        a = max_pi_ind.unsqueeze(-1).repeat(1, 1, C).unsqueeze(2)
        # (4, 58, 1, 136)
        lmk_pred = torch.gather(mdn_mu, 2, a)
        a = max_pi_ind.unsqueeze(-1).repeat(1, 1, 2).unsqueeze(2)
        sigmas = torch.gather(mdn_sigma, 2, a)
    elif args.mdn_sum:
        lmk_pred = torch.sum(mdn_mu * pi.unsqueeze(-1), 2)
        sigmas = torch.sum(mdn_sigma * pi.unsqueeze(-1), 2)
    
    lmk_pred = lmk_pred.squeeze(2)
    lmk_pred = lmk_pred.reshape(lmk_pred.shape[0], lmk_pred.shape[1], 68, 2)
    lmk_gt = lmk_gt.reshape(lmk_gt.shape[0], lmk_gt.shape[1], 68, 2)
    n1 = [1] + list(range(0, 16)) + [18] + list(range(17, 21)) + [23] + list(range(22, 26)) + \
                 [28] + list(range(27, 35)) + [41] + list(range(36, 41)) + [47] + list(range(42, 47)) + \
                 [59] + list(range(48, 59)) + [67] + list(range(60, 67))
    n2 = list(range(1, 17)) + [15] + list(range(18, 22)) + [20] + list(range(23, 27)) + [25] + \
                 list(range(28, 36)) + [34] + list(range(37, 42)) + [36] + list(range(43, 48)) + [42] + \
                 list(range(49, 60)) + [48] + list(range(61, 68)) + [60]
    V = lmk_pred

    V = V[:, :, :68]
    L_V = V - 0.5 * (V[:, :, n1, :] + V[:, :, n2, :])

    G = lmk_gt

    G = G[:, :, :68]
    L_G = G - 0.5 * (G[:, :, n1, :] + G[:, :, n2, :])
    loss_laplacian = torch.nn.functional.l1_loss(L_V, L_G)
    return loss_laplacian

def motion_loss(pi, outH, mdn_sigma, lmk_gt, args):
    pi_max, max_pi_ind = torch.max(pi, -1)
    M = args.M
    C = outH.shape[-1] // M

    mdn_mu = outH.view(outH.shape[0], outH.shape[1], M, 136)

    mdn_sigma = mdn_sigma.view(outH.shape[0], outH.shape[1], M, 2)

    if args.mdn_max:
        a = max_pi_ind.unsqueeze(-1).repeat(1, 1, C).unsqueeze(2)

        lmk_pred = torch.gather(mdn_mu, 2, a)
        a = max_pi_ind.unsqueeze(-1).repeat(1, 1, 2).unsqueeze(2)
        sigmas = torch.gather(mdn_sigma, 2, a)
    elif args.mdn_sum:
        lmk_pred = torch.sum(mdn_mu * pi.unsqueeze(-1), 2)
        sigmas = torch.sum(mdn_sigma * pi.unsqueeze(-1), 2)
    
    lmk_pred = lmk_pred.squeeze(2)

    pred_motion = lmk_pred[:, :-1] - lmk_pred[:, 1:]
    gt_motion = lmk_gt[:, :-1] - lmk_gt[:, 1:]
    loss = torch.nn.functional.l1_loss(pred_motion, gt_motion)
    return loss


def rot_loss(pi, outH, mdn_sigma, rot_quats_gt, rot_trans_gt,rot_quats_pred, rot_trans_pred, args):
    
    pi_max, max_pi_ind = torch.max(pi, -1)
    M = args.M
    C = outH.shape[-1] // M

    rot_quats_pred = rot_quats_pred.view(rot_quats_pred.shape[0], rot_quats_pred.shape[1], M, 1)
    
    rot_trans_pred = rot_trans_pred.view(rot_trans_pred.shape[0], rot_trans_pred.shape[1], M, 2)

    mdn_sigma = mdn_sigma.view(outH.shape[0], outH.shape[1], M, 2)

    if args.mdn_max:
        a_rot = max_pi_ind.unsqueeze(-1).repeat(1, 1, 1).unsqueeze(2)
        a_trans = max_pi_ind.unsqueeze(-1).repeat(1, 1, 2).unsqueeze(2)

        rot_quats_pred = torch.gather(rot_quats_pred, 2, a_rot)
        rot_trans_pred = torch.gather(rot_trans_pred, 2, a_trans)
        a = max_pi_ind.unsqueeze(-1).repeat(1, 1, 2).unsqueeze(2)
        sigmas = torch.gather(mdn_sigma, 2, a)
    elif args.mdn_sum:
        rot_quats_pred = torch.sum(rot_quats_pred * pi.unsqueeze(-1), 2)
        rot_trans_pred = torch.sum(rot_trans_pred * pi.unsqueeze(-1), 2)
        sigmas = torch.sum(mdn_sigma * pi.unsqueeze(-1), 2)
    
    rot_trans_pred = rot_trans_pred.squeeze(2)
    rot_quats_pred = rot_quats_pred.squeeze(2)
    
    gt = torch.cat([rot_quats_gt.unsqueeze(2), rot_trans_gt[:, :, :, 2]], dim=-1)
    loss = torch.nn.functional.l1_loss(torch.cat([rot_quats_pred, rot_trans_pred], dim=-1), gt)
    return loss
    
def close_face_lip(fl):

    fl = fl.reshape(fl.shape[0], fl.shape[1], 68, 2)
    from util.geo_math import area_of_polygon
    ids = []
    for facelandmark in fl:
        min_area_lip, idx = 999, 0
        for i, fls in enumerate(facelandmark):
            area_of_mouth = area_of_polygon(fls[list(range(60, 68)), 0:2])
            if (area_of_mouth < min_area_lip):
                min_area_lip = area_of_mouth
                idx = i
        ids.append(idx)
    return ids


def set_path(args):
    if args.resume: exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = 'log_{args.prefix}/{args.dataset}--\
{args.net}_bs{args.batch_size}_\
lr{0}_wd{args.wd}_train-{args.train_what}{1}'.format(
                    args.old_lr if args.old_lr is not None else args.lr, \
                    '_pt='+args.pretrain.replace('/','-') if args.pretrain else '', \
                    args=args)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    video_path = os.path.join(exp_path, 'video')
    lmk_path = os.path.join(exp_path, 'lmk')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    if not os.path.exists(video_path): os.makedirs(video_path)
    if not os.path.exists(lmk_path): os.makedirs(lmk_path)
    return img_path, model_path, video_path, lmk_path


def MultiStepLR_Restart_Multiplier(epoch, gamma=0.1, step=[10,15,20], repeat=3):
    '''return the multipier for LambdaLR,
    0  <= ep < 10: gamma^0
    10 <= ep < 15: gamma^1
    15 <= ep < 20: gamma^2
    20 <= ep < 30: gamma^0 ... repeat 3 cycles and then keep gamma^2'''
    max_step = max(step)
    effective_epoch = epoch % max_step
    if epoch // max_step >= repeat:
        exp = len(step) - 1
    else:
        exp = len([i for i in step if effective_epoch>=i])
    return gamma ** exp


def video_display(pi, outH, mdn_sigma, lmk_gt, audio, data_loader_length, epoch, idx, video_name, args, pos_std=None):
    pi_max, max_pi_ind = torch.max(pi, -1)
    M = args.M
    C = outH.shape[-1] // M
    B = outH.shape[0]
    SEQ_LEN = outH.shape[1]

    outH = outH.view(outH.shape[0], outH.shape[1], M, 136 + 1 + 2)
    mdn_mu = outH[:, :, :, :136]

    mdn_sigma = mdn_sigma.view(outH.shape[0], outH.shape[1], M, 1)

    if args.mdn_max:
        a = max_pi_ind.unsqueeze(-1).repeat(1, 1, C).unsqueeze(2)
        pred = torch.gather(outH, 2, a)
        a = max_pi_ind.unsqueeze(-1).repeat(1, 1, 1).unsqueeze(2)
        sigmas = torch.gather(mdn_sigma, 2, a)
    elif args.mdn_sum:
        lmk_pred = torch.sum(outH * pi.unsqueeze(-1), 2)
        sigmas = torch.sum(mdn_sigma * pi.unsqueeze(-1), 2)

    lmk_pred = pred[:, :, :, :136]
    pose_pred = pred[:, :, :, 136:137]
    trans_pred = pred[:, :, :, 137:139]

    trans_pred = trans_pred.squeeze(2)
    pose_pred = pose_pred.squeeze(2)
    
    

    random_clip_index = np.random.permutation(data_loader_length)[0:args.random_clip_num]

    def save_fls_av(fake_fls_list, postfix='', ifsmooth=True):

        fake_fls_np = fake_fls_list
        v_name = video_name[0]
        filename = 'fake_fls_{}_{}_{}.txt'.format(epoch, v_name, postfix)
        np.savetxt(
                    os.path.join(args.lmk_path, filename),
                    fake_fls_np, fmt='%.6f')
        au = audio[0]
        audio_filename = os.path.join(args.video_path, '1.wav')
        from util.vis import Vis_old
        Vis_old(run_name=None, pred_fl_filename=filename, audio_filename=audio_filename,
                        fps=25, av_name='e{:04d}_{}'.format(epoch, postfix + '_' + v_name),
                        root_dir=args.lmk_path, ifsmooth=ifsmooth, audio=au)

    pose_pred = pose_pred.detach().cpu().numpy()
    quat = pose_pred
    quat = pose_pred.reshape(quat.shape[0] * quat.shape[1], quat.shape[2])
    tmp = []
    for q in quat:
        tmp.append(np.array([[[math.cos(q[0]), -math.sin(q[0])],
                                    [math.sin(q[0]), math.cos(q[0])]]]))
    quat = np.concatenate(tmp, axis=0)

    trans = trans_pred.detach().cpu().numpy()
    trans = trans.reshape(trans.shape[0] * trans.shape[1], 2)
    lmk_pred = lmk_pred.squeeze(2)
    lmk_pred = lmk_pred.reshape(lmk_pred.shape[0] * lmk_pred.shape[1], 68, 2)
    lmk_pred = lmk_pred.data.cpu().numpy()
    for k in range(lmk_pred.shape[0]):
        lmk_pred[k] = np.dot(quat[k].T,
                                    (lmk_pred[k] - trans[k:k + 1]).T).T
    lmk_pred = lmk_pred.reshape(B, SEQ_LEN, 1, 68 * 2)
    
    pos_std = pos_std.reshape((-1, 2, 3))
    pos_std= pos_std.data.cpu().numpy()
    lmk_gt = lmk_gt.reshape(-1, 68, 2)
    lmk_gt = lmk_gt.data.cpu().numpy()
    for k in range(lmk_gt.shape[0]):
        lmk_gt[k] = np.dot(pos_std[k, :2, :2].T,
                            (lmk_gt[k] - pos_std[k, :, 2].T).T).T
    lmk_gt = lmk_gt.reshape(B, SEQ_LEN, 68 * 2)
    
    
    
    # and not is_training
    if (args.show_animation):

        print('show animation ....')

        lmk_pred = lmk_pred.squeeze(2)[0]

        lmk_pred = savgol_filter(lmk_pred, 7, 3, axis=0)
        lmk_gt = lmk_gt[0]
        v_name = video_name[0]
        aud = audio[0]
        save_fls_av(lmk_pred, 'pred_{}'.format(idx), ifsmooth=True)
        save_fls_av(lmk_gt, 'std_{}'.format(idx), ifsmooth=True)
        from util.vis import Vis_comp
        Vis_comp(run_name=None,
                     pred1='fake_fls_{}_{}_{}.txt'.format(epoch, v_name, 'pred_{}'.format(idx)),
                     pred2='fake_fls_{}_{}_{}.txt'.format(epoch, v_name, 'std_{}'.format(idx)),
                     audio_filename=os.path.join(args.video_path, '1.wav'),
                     fps=25, av_name='e{:04d}_{}_{}'.format(epoch, v_name, 'comp_{}'.format(idx)),
                     postfix='comp_{}'.format(idx), root_dir=args.lmk_path, ifsmooth=False, audio=aud)



def compare_landmarks(pi, outH, mdn_sigma, lmk_gt, args, pos_std=None):
    pi_max, max_pi_ind = torch.max(pi, -1)
    M = args.M
    C = outH.shape[-1] // M
    B = outH.shape[0]
    SEQ_LEN = outH.shape[1]

    outH = outH.view(outH.shape[0], outH.shape[1], M, 136 + 1 + 2)
    mdn_mu = outH[:, :, :, :136]

    mdn_sigma = mdn_sigma.view(outH.shape[0], outH.shape[1], M, 1)

    if args.mdn_max:
        a = max_pi_ind.unsqueeze(-1).repeat(1, 1, C).unsqueeze(2)
        pred = torch.gather(outH, 2, a)
        a = max_pi_ind.unsqueeze(-1).repeat(1, 1, 1).unsqueeze(2)
        sigmas = torch.gather(mdn_sigma, 2, a)
    elif args.mdn_sum:
        lmk_pred = torch.sum(outH * pi.unsqueeze(-1), 2)
        sigmas = torch.sum(mdn_sigma * pi.unsqueeze(-1), 2)
    lmk_pred = pred[:, :, :, :136]
    pose_pred = pred[:, :, :, 136:137]
    trans_pred = pred[:, :, :, 137:139]

    
    trans_pred = trans_pred.squeeze(2)
    pose_pred = pose_pred.squeeze(2)

    pose_pred = pose_pred.detach().cpu().numpy()
    quat = pose_pred
    quat = pose_pred.reshape(quat.shape[0] * quat.shape[1], quat.shape[2])
    tmp = []
    for q in quat:
        tmp.append(np.array([[[math.cos(q[0]), -math.sin(q[0])],
                                    [math.sin(q[0]), math.cos(q[0])]]]))
    quat = np.concatenate(tmp, axis=0)

    trans = trans_pred.detach().cpu().numpy()
    trans = trans.reshape(trans.shape[0] * trans.shape[1], 2)
    lmk_pred = lmk_pred.squeeze(2)
    lmk_pred = lmk_pred.reshape(lmk_pred.shape[0] * lmk_pred.shape[1], 68, 2)
    lmk_pred = lmk_pred.data.cpu().numpy()
    for k in range(lmk_pred.shape[0]):
        lmk_pred[k] = np.dot(quat[k].T,
                                    (lmk_pred[k] - trans[k:k + 1]).T).T
    lmk_pred = lmk_pred.reshape(B, SEQ_LEN, 1, 68 * 2)
    
    pos_std = pos_std.reshape((-1, 2, 3))
    pos_std= pos_std.data.cpu().numpy()
    lmk_gt = lmk_gt.reshape(-1, 68, 2)
    lmk_gt = lmk_gt.data.cpu().numpy()
    for k in range(lmk_gt.shape[0]):

        lmk_gt[k] = np.dot(pos_std[k, :2, :2].T,
                            (lmk_gt[k] - pos_std[k, :, 2].T).T).T
    lmk_gt = lmk_gt.reshape(B, SEQ_LEN, 68 * 2)
    

    lmk_pred = lmk_pred.squeeze(2)
    lmk_gt = lmk_gt
    rp = lmk_pred.reshape(lmk_pred.shape[0] * lmk_pred.shape[1], 68, 2)
    fp = lmk_gt.reshape(lmk_gt.shape[0] * lmk_gt.shape[1], 68, 2)
    distances = []
    for inx in range(len(rp)):
        rp_mouth = rp[inx, 48:68]
        fp_mouth = fp[inx, 48:68]

        rp_mouth_land = rp_mouth.copy()

        fp_mouth_land = fp_mouth.copy()

        dis = (rp_mouth_land-fp_mouth_land)**2
        dis = np.sum(dis,axis=1)
        dis = np.sqrt(dis)

        dis = np.mean(dis, axis=0)
        distances.append(dis)

    average_distance = sum(distances) / len(rp)
    
    return average_distance


def analysis(pi, outH, mdn_sigma, lmk_gt, args,  dict_ct, dict_rot, pos_std=None):
    
    pi_max, max_pi_ind = torch.max(pi, -1)
    max_pi_ind_np = max_pi_ind.detach().cpu().numpy();
    lmk_gt_np = lmk_gt.detach().cpu().numpy()
    for bt, bt_data in zip(max_pi_ind_np, lmk_gt_np):
        for seq, seq_data in zip(bt, bt_data):
            dict_ct[seq] = dict_ct[seq] + 1
            dict_rot[seq] += seq_data[136]


def analysis_speaker(pi, pid_gt, dict_speaker):
    
    pi_max, max_pi_ind = torch.max(pi, -1)
    max_pi_ind_np = max_pi_ind.detach().cpu().numpy();
    pid_gt_np = pid_gt.detach().cpu().numpy()
    for bt, bt_data in zip(max_pi_ind_np, pid_gt_np):
        for seq, seq_data in zip(bt, bt_data):
            dict_speaker[seq][seq_data] =  dict_speaker[seq][seq_data] + 1


def video_display_model(pi, outH, mdn_sigma, lmk_gt, audio, data_loader_length, epoch, idx, video_name, args, pos_std=None):
    pi_max, max_pi_ind = torch.max(pi, -1)
    M = args.M
    C = outH.shape[-1] // M
    B = outH.shape[0]
    SEQ_LEN = outH.shape[1]

    outH = outH.view(outH.shape[0], outH.shape[1], M, 136 + 1 + 2)
    mdn_mu = outH[:, :, :, :136]

    mdn_sigma = mdn_sigma.view(outH.shape[0], outH.shape[1], M, 1)
    lmk_pred = outH[:, :, :, :136]
    pose_pred = outH[:, :, :, 136:137]
    trans_pred = outH[:, :, :, 137:139]
    
    for i in range(M):
        def save_fls_av(fake_fls_list, postfix='', ifsmooth=True):

            fake_fls_np = fake_fls_list
            v_name = video_name[0]
            filename = 'fake_fls_{}_{}_{}.txt'.format(epoch, v_name, postfix)
            np.savetxt(
                        os.path.join(args.lmk_path, filename),
                        fake_fls_np, fmt='%.6f')
            au = audio[0]
            audio_filename = os.path.join(args.video_path, '1.wav')
            from util.vis import Vis_old
            Vis_old(run_name=None, pred_fl_filename=filename, audio_filename=audio_filename,
                            fps=15, av_name='e{:04d}_{}'.format(epoch, postfix + '_' + v_name),
                            root_dir=args.lmk_path, ifsmooth=ifsmooth, audio=au)

        pose_pred_i = pose_pred[:, :, i, :].detach().cpu().numpy()
        quat = pose_pred_i
        quat = pose_pred_i.reshape(quat.shape[0] * quat.shape[1], quat.shape[2])
        tmp = []
        for q in quat:
            tmp.append(np.array([[[math.cos(q[0]), -math.sin(q[0])],
                                        [math.sin(q[0]), math.cos(q[0])]]]))
        quat = np.concatenate(tmp, axis=0)
        trans_i = trans_pred[:, :, i, :].detach().cpu().numpy()
        trans_i = trans_i.reshape(trans_i.shape[0] * trans_i.shape[1], 2)
        lmk_pred_i = lmk_pred[:, :, i, :]
        lmk_pred_i = lmk_pred_i.reshape(lmk_pred_i.shape[0] * lmk_pred_i.shape[1], 68, 2)
        lmk_pred_i = lmk_pred_i.data.cpu().numpy()
        for k in range(lmk_pred_i.shape[0]):
            lmk_pred_i[k] = np.dot(quat[k].T,
                                        (lmk_pred_i[k] - trans_i[k:k + 1]).T).T
        lmk_pred_i = lmk_pred_i.reshape(B, SEQ_LEN, 1, 68 * 2)
        
        # and not is_training
        if (args.show_animation):
            print('show animation ....')

            lmk_pred_i = lmk_pred_i.squeeze(2)[0]

            lmk_pred_i = savgol_filter(lmk_pred_i, 7, 3, axis=0)
            v_name = video_name[0]
            aud = audio[0]
            save_fls_av(lmk_pred_i, 'mode_{}_pred_{}'.format(i, idx), ifsmooth=True)


if __name__ == '__main__':
    args = parse_args()
    main(args)
