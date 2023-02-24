import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')
from backbone.select_backbone3D import select_resnet
from backbone.audio_net import WavEncoder
from utils.utils import calc_topk_accuracy

class LossScale(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0):
        super(LossScale, self).__init__()
        
        self.wI = nn.Parameter(torch.tensor(init_w))
        self.bI = nn.Parameter(torch.tensor(init_b))

        self.wC = nn.Parameter(torch.tensor(init_w))
        self.bC = nn.Parameter(torch.tensor(init_b))



class MemDPC_BD(nn.Module):
    '''MemDPC with bi-directional RNN'''

    def __init__(self,
                 is_bidirectional,
                 dropout=0.0,
                 network='resnet18',
                 mem_size=1024):
        super(MemDPC_BD, self).__init__()
        print('Using MemDPC-BiDirectional model with {} and mem_size {}' \
              .format(network, mem_size))
        self.mem_size = mem_size
        self.is_bidirectional = is_bidirectional
        self.tgt_dict = {}

        self.backbone, self.param = select_resnet(network)
        self.param['num_layers'] = 1  # param for GRU
#         self.param['rnn_input_size'] = 512 # param for GRU
#         self.param['hidden_size'] = self.param['rnn_input_size']  # param for GRU
        self.param['rnn_input_size'] = 256 # param for GRU
        self.param['hidden_size'] = 512  # param for GRU
        self.param['membanks_size'] = mem_size
        
        self.__L__ = LossScale()
        
        self.mb_content = torch.nn.Parameter(torch.randn(self.param['membanks_size'], self.param['feature_size']))
        self.mb_spk = torch.nn.Parameter(torch.randn(self.param['membanks_size'], self.param['feature_size']))
        print('MEM Bank has size %dx%d' % (self.param['membanks_size'], self.param['feature_size']))

        # bi-directional RNN
        self.GRU = nn.GRU(input_size=self.param['rnn_input_size'],
                          hidden_size=self.param['hidden_size'],
                          num_layers=self.param['num_layers'],
#                           dropout=dropout,
                          batch_first=True,
                          bidirectional=True if is_bidirectional else False
                          )
        # image Encoder
        # self.param['feature_size']
        self.imgEncoder_content = nn.Sequential(nn.Conv3d(256, self.param['feature_size'],
                                                  kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                                        nn.BatchNorm3d(self.param['feature_size']),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Conv3d(self.param['feature_size'], self.param['feature_size'],
                                                  kernel_size=(1, 4, 4), stride=1, padding=0),
                                        nn.AvgPool3d((2, 1, 1), stride=1)
                                        )
        
        
        self.imgEncoder_spk = nn.Sequential(nn.Conv3d(256, self.param['feature_size'],
                                          kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                                nn.BatchNorm3d(self.param['feature_size']),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Conv3d(self.param['feature_size'], self.param['feature_size'],
                                          kernel_size=(1, 4, 4), stride=1, padding=0),
                                nn.AvgPool3d((2, 1, 1), stride=1)
                                )
        # audio Encoder
        self.audioEncoder = WavEncoder()
        if is_bidirectional:
            self.network_pred_content = nn.Sequential(
                nn.Linear(self.param['hidden_size'] * 2, self.param['feature_size']),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.param['feature_size'], self.param['membanks_size'])
            )
            
            self.network_pred_spk = nn.Sequential(
                nn.Linear(self.param['hidden_size'] * 2, self.param['feature_size']),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.param['feature_size'], self.param['membanks_size'])
            )
        else:
            self.network_pred_content = nn.Sequential(
                nn.Linear(self.param['hidden_size'], self.param['feature_size']),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.param['feature_size'], self.param['membanks_size'])
            )
            
            self.network_pred_spk = nn.Sequential(
                nn.Linear(self.param['hidden_size'], self.param['feature_size']),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.param['feature_size'], self.param['membanks_size'])
            )
        for m in self.network_pred_content.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        for m in self.network_pred_spk.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.relu = nn.ReLU(inplace=False)
        self.ce_loss = nn.CrossEntropyLoss()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._initialize_weights(self.GRU)


    def sync_loss(self,out_v,out_a,criterion):
        out_a = out_a.permute(0, 2, 1).contiguous()
        out_v = out_v.permute(0, 2, 1).contiguous()
        batch_size  = out_a.size()[0]
        time_size   = out_a.size()[2]

        label       = torch.arange(time_size).cuda()

        nloss = 0
        prec1 = 0
        prec5 = 0

        for ii in range(0,batch_size):
            ft_v    = out_v[[ii],:,:].transpose(2,0)
            ft_a    = out_a[[ii],:,:].transpose(2,0)
            output  = F.cosine_similarity(ft_v.expand(-1,-1,time_size),ft_a.expand(-1,-1,time_size).transpose(0,2)) * self.__L__.wC + self.__L__.bC
            p1, p5  = calc_topk_accuracy(output.detach().cpu(), label.detach().cpu(), topk=(1, 5))

            nloss += criterion(output, label)
            prec1 += p1
            prec5 += p5
            

        nloss       = nloss / batch_size
        prec1       = prec1 / batch_size
        prec5       = prec5 / batch_size

        return nloss, prec1, prec5

    def get_loss(self, pred, gt, B, SEQ_LEN, feature_size):
        # pred: B,SEQ_LEN,256
        # GT: B, SEQ_LEN, 256

        pred = pred.view(B * SEQ_LEN, feature_size)
        gt = gt.contiguous().view(feature_size, B * SEQ_LEN)
        score = torch.matmul(F.normalize(pred, p=2, dim=1),
                             F.normalize(gt, p=2, dim=0))
        # 设置tem
        score = score / 0.05
        if SEQ_LEN not in self.tgt_dict:
            self.tgt_dict[SEQ_LEN] = torch.arange(B * SEQ_LEN)
        tgt = self.tgt_dict[SEQ_LEN].to(score.device)
        loss = self.ce_loss(score, tgt)
        top1, top5 = calc_topk_accuracy(score, tgt, (1, 3))
        return loss, top1, top5
    """
    audio:(batch_size, seq_len, w, h)
    img:(batch_size, seq_len, C, w, h)
    """
    def forward(self, audio, img, add_noise=False, i=0):

        (B, SEQ_LEN, *_) = audio.shape
        # image
        img = img.view(B * SEQ_LEN, img.shape[2], img.shape[3], img.shape[4], img.shape[5])
        img_feature = self.backbone(img)
        #####content####
        img_feature_content = self.imgEncoder_content(img_feature)
        img_feature_content = img_feature_content.view(B, SEQ_LEN,
                                       img_feature_content.shape[1] * img_feature_content.shape[2] * img_feature_content.shape[3]) # before ReLU, (-inf, +

        gt_content = img_feature_content
        
        
        #####speak####
        img_feature_spk = self.imgEncoder_spk(img_feature)
        img_feature_spk = img_feature_spk.view(B, SEQ_LEN,
                                       img_feature_spk.shape[1] * img_feature_spk.shape[2] * img_feature_spk.shape[3]) # before ReLU, (-inf, +
        
#         gt_spk = img_feature_spk.permute(2, 0, 1).contiguous()
        gt_spk = img_feature_spk
        

        # audio
        gru_input = self.audioEncoder(audio)
        if add_noise:
            au_noise = torch.tensor(torch.randn(gru_input.shape) * 0.01, requires_grad=False, dtype=torch.float).cuda()
            gru_input = gru_input + au_noise
        gru_input = gru_input.view(B, SEQ_LEN,
                                           gru_input.shape[1] * gru_input.shape[2] * gru_input.shape[3])
        hidden_state = torch.zeros(2 * self.param['num_layers'] if self.is_bidirectional else 1 * self.param['num_layers'],
                             B, self.param['hidden_size'], device=self.device
                             )
        output, _ = self.GRU(gru_input, hidden_state)


        losses = []  # all loss
        acc = []  # all acc
        loss = 0
        # forward MemDPC
        
        ####content#####
        pd_tmp_content = self.network_pred_content(output)
        pd_tmp_content = F.softmax(pd_tmp_content, dim=-1)
        pd_tmp_content = torch.einsum('bnm,mc->bnc', pd_tmp_content, self.mb_content)
        
        ####speaker#####
        pd_tmp_spk = self.network_pred_spk(output)
        pd_tmp_spk = F.softmax(pd_tmp_spk, dim=-1)
        pd_tmp_spk = torch.einsum('bnm,mc->bnc', pd_tmp_spk, self.mb_spk)
        
        
        # sync loss and accuracy
        nloss_sy, p1s, p5s = self.sync_loss(gt_content,pd_tmp_content, self.ce_loss)
        
        ri          = random.randint(0,SEQ_LEN-1)
        
        
        out_AA      = torch.mean(pd_tmp_spk,1,keepdim=True);
        out_VA      = gt_spk[:, [ri], :]
        
        out_AA = out_AA.permute(0, 2, 1).contiguous()
        out_VA = out_VA.permute(0, 2, 1).contiguous()
        
        # identity loss and accuracy
        idoutput = F.cosine_similarity(out_VA.expand(-1,-1,B),out_AA.expand(-1,-1,B).transpose(0,2)) * self.__L__.wI + self.__L__.bI
        
        label_id = torch.arange(B).cuda()
        
        
        nloss_id    = self.ce_loss(idoutput, label_id)
            
        p1i, p5i    = calc_topk_accuracy(idoutput.detach().cpu(), label_id.detach().cpu(), topk=(1, 3))
        
        
        
        # ==================== Divergence Loss ====================

        nloss = nloss_sy + nloss_id
        

        losses.append(nloss.data.unsqueeze(0))
        acc.append(torch.stack([p1s, p5s, p1i, p5i], 0).unsqueeze(0))

        return nloss, losses, acc

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
#                 nn.init.orthogonal_(param, 0.1)
                nn.init.xavier_normal_(param, 1)
