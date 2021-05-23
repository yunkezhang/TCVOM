import torch
import torch.nn as nn
import torch.nn.functional as F
from models.VMN.VMN_model import FeatureAggregationModule

class DIMEncoder(nn.Module):
    def __init__(self, input_chn):
        super(DIMEncoder, self).__init__()
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

        self.conv6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)

    def forward(self, x):
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
            
        x6 = F.relu(self.conv6(x5p))
        return [idx1p, idx2p, idx3p, idx4p, idx5p, x6]

class DIMDecoder(nn.Module):
    def __init__(self, reduction, window, freeze_backbone):
        super(DIMDecoder, self).__init__()
        # decoding
        self.freeze_backbone = freeze_backbone
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

        self.fam = FeatureAggregationModule(256, reduction, window)
        #self.fam_conv = nn.Conv2d(256 + 256 // reduction * 2, 256, kernel_size=1, padding=0)

    def train(self, mode):
        super().train(mode)
        if self.freeze_backbone:
            print ('Set DIM decoder feature extraction part in eval() mode.')
            self.dconv6.eval()
            self.dconv5.eval()
            self.dconv4.eval()

    def forward(self, inputs, extract_feature, x=None, xb=None, xf=None, mask=None):
        if extract_feature:
            idx1p, idx2p, idx3p, idx4p, idx5p, x6 = inputs
            x6d = F.relu(self.dconv6(x6))

            x5d = self.unpool5(x6d, indices=idx5p)  # OS=16
            x5d = F.relu(self.dconv5(x5d))

            x4d = self.unpool4(x5d, indices=idx4p)  # OS=8
            x4d = F.relu(self.dconv4(x4d))
            return x4d
        else:
            idx1p, idx2p, idx3p, idx4p, idx5p, x6 = inputs
            x, attb, attf, mask = self.fam(x, xb, xf, mask)
            #x4d = self.fam_conv(torch.cat([x, xb, xf], dim=1))
            #x4d = F.relu(x)

            x3d = self.unpool3(x, indices=idx3p)  # OS=4
            x3d = F.relu(self.dconv3(x3d))

            x2d = self.unpool2(x3d, indices=idx2p)  # OS=2
            x2d = F.relu(self.dconv2(x2d))

            x1d = self.unpool1(x2d, indices=idx1p)  # OS=1
            x1d = F.relu(self.dconv1(x1d))

            xpred = self.alpha_pred(x1d).clamp(0, 1)
            return xpred, attb, attf, mask