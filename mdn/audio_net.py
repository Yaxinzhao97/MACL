import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.LeakyReLU(0.2)
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class WavEncoder(nn.Module):

    def __init__(self):
        super(WavEncoder, self).__init__()
        self.encoder = nn.Sequential(
            Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True), # (16, 20, 80)
#             nn.Dropout(0.5),
            Conv2d(16, 32, kernel_size=3, stride=(2, 4), padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), # (32, 10, 20)
#             nn.Dropout(0.5),
            Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), # (64, 5, 10)
#             nn.Dropout(0.5),
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), # (128, 3, 5)
#             nn.Dropout(0.5),
            Conv2d(128, 256, kernel_size=(3, 5), stride=1, padding=0),

        )

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = x.unsqueeze(2)
        b, n, c, w, h = x.shape
        x = x.view(b * n, c, w, h)

        audio_feature = self.encoder(x)
        return audio_feature
    
