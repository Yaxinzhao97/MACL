import torch
import torch.nn as nn
import torch.nn.functional as F
from audio_net import WavEncoder
from mixnetwork_cs import MixNet


class ModelLmk(nn.Module):
    def __init__(self,
                 is_bidirectional=True,
                 dropout=0.0,
                 M =3,
                 mem_size=256,
                 train_what='all'
                 ):
        super(ModelLmk, self).__init__()

        self.train_what = train_what
        self.is_bidirectional = is_bidirectional
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.param = {}
        self.param['num_layers'] = 1
        self.param['rnn_input_size'] = 256
        self.param['hidden_size'] = 128
        self.param['membanks_size'] = mem_size
        self.param['feature_size'] = 128
        self.mb_content = torch.nn.Parameter(torch.randn(self.param['membanks_size'], self.param['feature_size']))
        self.mb_spk = torch.nn.Parameter(torch.randn(self.param['membanks_size'], self.param['feature_size']))
        print('MEM Bank has size %dx%d' % (self.param['membanks_size'], self.param['feature_size']))

        # bi-directional RNN
        self.GRU = nn.GRU(input_size=self.param['rnn_input_size'],
                          hidden_size=self.param['hidden_size'],
                          num_layers=self.param['num_layers'],
                          batch_first=True,
                          bidirectional=True if is_bidirectional else False
                          )
        
        
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
                nn.Linear(self.param['feature_size'], self.param['feature_size'])
            )
            
            self.network_pred_spk = nn.Sequential(
                nn.Linear(self.param['hidden_size'], self.param['feature_size']),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(self.param['feature_size'], self.param['feature_size'])
            )
        
        
        # audio Encoder
        self.audioEncoder = WavEncoder()
        self._initialize_weights(self.GRU)

        self.final_bn_content = nn.BatchNorm1d(self.param['feature_size'])
        self.final_bn_content.weight.data.fill_(1)
        self.final_bn_content.bias.data.zero_()
        
        self.final_bn_spk = nn.BatchNorm1d(self.param['feature_size'])
        self.final_bn_spk.weight.data.fill_(1)
        self.final_bn_spk.bias.data.zero_()


        # landmark
        self.lmkNet = MixNet(self.param['feature_size'], M)

    def forward(self, audio, lmk, add_noise=False):

        (B, SEQ_LEN, *_) = audio.shape
        enable_grad = self.train_what != 'last'
        with torch.set_grad_enabled(enable_grad):
            # audio
            gru_input = self.audioEncoder(audio)
            if add_noise:
                au_noise = torch.tensor(torch.randn(gru_input.shape) * 0.01, requires_grad=False, dtype=torch.float).cuda()
                gru_input = gru_input + au_noise
            gru_input = gru_input.view(B, SEQ_LEN,
                                       gru_input.shape[1] * gru_input.shape[2] * gru_input.shape[3])

            hidden_state = torch.zeros(
                2 * self.param['num_layers'] if self.is_bidirectional else 1 * self.param['num_layers'],
                B, self.param['hidden_size'], device=self.device
                )
            output, _ = self.GRU(gru_input, hidden_state)
            ##########memorybank#####################################
            
            ####content#####
            pd_tmp_content = self.network_pred_content(output)
            pd_tmp_content = F.softmax(pd_tmp_content, dim=-1)
            pd_tmp_content = torch.einsum('bnm,mc->bnc', pd_tmp_content, self.mb_content)
        
            ####speaker#####
            pd_tmp_spk = self.network_pred_spk(output)
            pd_tmp_spk = F.softmax(pd_tmp_spk, dim=-1)
            pd_tmp_spk = torch.einsum('bnm,mc->bnc', pd_tmp_spk, self.mb_spk)
            
            

        out_content = self.final_bn_content(pd_tmp_content.transpose(-1, -2)).transpose(-1, -2) # [B,S,C] -> [B,C,S] -> BN() -> [B,S,C], because BN operates on id=1 channel.
        out_spk = self.final_bn_spk(pd_tmp_spk.transpose(-1, -2)).transpose(-1, -2) # [B,S,C] -> [B,C,S] -> BN() -> [B,S,C], because BN operates on id=1 channel.

        return self.lmkNet(out_spk, out_content, lmk)

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

