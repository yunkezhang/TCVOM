"""
Implementation of Deep Image Matting @ CVPR2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

class DeepMatting(nn.Module):
    def __init__(self, input_chn, output_chn=1, build_decoder=True, freeze_bn=False, freeze_dropout=False, alpha_only=True):
        super(DeepMatting, self).__init__()
        self.alpha_only = alpha_only
        self.input_chn = input_chn
        self.build_decoder = build_decoder
        self.freeze_bn = freeze_bn
        self.freeze_dropout = freeze_dropout
        # encoding
        self.conv11 = nn.Conv2d(input_chn, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 2), stride=2, return_indices=True)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((2, 2), stride=2, return_indices=True)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d((2, 2), stride=2, return_indices=True)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d((2, 2), stride=2, return_indices=True)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d((2, 2), stride=2, return_indices=True)

        if self.build_decoder:
            self.conv6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)

            # decoding
            self.dconv6 = nn.Conv2d(4096, 512, kernel_size=1, padding=0)

            self.unpool5 = nn.MaxUnpool2d((2, 2), stride=2)
            self.dconv5 = nn.Conv2d(512, 512, kernel_size=5, padding=2)

            self.unpool4 = nn.MaxUnpool2d((2, 2), stride=2)
            self.dconv4 = nn.Conv2d(512, 256, kernel_size=5, padding=2)

            self.unpool3 = nn.MaxUnpool2d((2, 2), stride=2)
            self.dconv3 = nn.Conv2d(256, 128, kernel_size=5, padding=2)

            self.unpool2 = nn.MaxUnpool2d((2, 2), stride=2)
            self.dconv2 = nn.Conv2d(128, 64, kernel_size=5, padding=2)

            self.unpool1 = nn.MaxUnpool2d((2, 2), stride=2)
            self.dconv1 = nn.Conv2d(64, 64, kernel_size=5, padding=2)

            self.alpha_pred = nn.Conv2d(64, 1, kernel_size=5, padding=2)

    def forward(self, x, **kwargs):
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, idx1p = self.pool1(x12)

        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, idx2p = self.pool2(x22)

        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, idx3p = self.pool3(x33)

        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, idx4p = self.pool4(x43)

        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, idx5p = self.pool5(x53)


        if not self.build_decoder:
            return [x1p, x2p, x3p, x4p, x5p]
        x6 = F.relu(self.conv6(x5p))

        x6d = F.relu(self.dconv6(x6))

        x5d = self.unpool5(x6d, indices=idx5p)
        x5d = F.relu(self.dconv5(x5d))

        x4d = self.unpool4(x5d, indices=idx4p)
        x4d = F.relu(self.dconv4(x4d))

        x3d = self.unpool3(x4d, indices=idx3p)
        x3d = F.relu(self.dconv3(x3d))

        x2d = self.unpool2(x3d, indices=idx2p)
        x2d = F.relu(self.dconv2(x2d))

        x1d = self.unpool1(x2d, indices=idx1p)
        x1d = F.relu(self.dconv1(x1d))

        xpred = self.alpha_pred(x1d).clamp(0, 1)
        if self.alpha_only:
            return xpred
        else:
            return [x1p, x2p, x3p, x4p, x5p], xpred


def DIM_VGG(build_decoder=True, alpha_only=True):
    model = DeepMatting(input_chn=4, build_decoder=build_decoder, alpha_only=True)
    return model