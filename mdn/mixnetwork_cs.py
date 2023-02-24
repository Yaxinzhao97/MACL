import torch
import torch.nn as nn
import torch.nn.functional as F

LMK_DIM = 136
LMK_DIM_CONTENT = 128
LMK_DIM_SPK = 128
LMK_DIM_ENC = 64


class MixNet(nn.Module):
    def __init__(self, input_dim, M, dropout=0.8):
        super(MixNet, self).__init__()

        
        self.outheader = nn.Sequential(nn.Dropout(dropout), nn.Linear(LMK_DIM_CONTENT + LMK_DIM_SPK + LMK_DIM_ENC, (LMK_DIM + 1 + 2) * M))
        self.sigmaheader = nn.Sequential(nn.Dropout(dropout), nn.Linear(LMK_DIM_CONTENT + LMK_DIM_SPK + LMK_DIM_ENC, M))
        self.logitsheader =  nn.Sequential(nn.Dropout(dropout), nn.Linear(LMK_DIM_CONTENT + LMK_DIM_SPK + LMK_DIM_ENC, M))
        self.lmk_encoder = nn.Linear(LMK_DIM, LMK_DIM_ENC)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm1d(LMK_DIM_ENC)
        self.mid = nn.Linear(LMK_DIM_CONTENT + LMK_DIM_SPK + LMK_DIM_ENC, LMK_DIM_CONTENT + LMK_DIM_SPK + LMK_DIM_ENC)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_embedding_content, input_embedding_spk, lmk):

        lmk = self.lmk_encoder(lmk)
        input_embedding = torch.cat((input_embedding_content, input_embedding_spk, lmk), dim=-1)
        input_embedding = self.act(input_embedding)
        input_embedding = self.mid(input_embedding)
        outH = self.outheader(input_embedding)
        sigma = self.sigmaheader(input_embedding)
        logits = self.logitsheader(input_embedding)
        return outH, sigma, logits

def weights_init_he(m):
    """Initialize the given linear layer using He initialization."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.in_features * m.out_features
        m.weight.data.normal_(0.0, np.sqrt(2 / n))
        m.bias.data.fill_(0)

def weights_init_zeros(m):
    """Initialize the given linear layer with zeroes."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.zeros_(m.bias.data)
        nn.init.zeros_(m.weight.data)

