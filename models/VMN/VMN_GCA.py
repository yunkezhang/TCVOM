import torch
import torch.nn as nn
from models.GCA.ops import GuidedCxtAtten
from models.GCA.decoders.resnet_dec import ResNet_D_Dec, BasicBlock
from models.GCA import encoders
from models.VMN.VMN_model import FeatureAggregationModule

class ResGuidedCxtAtten_FAM_Dec(ResNet_D_Dec):

    def __init__(self, reduction, window, block=BasicBlock, layers=[2, 3, 3, 2], \
        norm_layer=None, large_kernel=False, freeze_backbone=False):
        super(ResGuidedCxtAtten_FAM_Dec, self).__init__(block, layers, norm_layer, \
            large_kernel=large_kernel, layer_multi=[1, 1, reduction])#1 + 2.0 / reduction])
        self.gca = GuidedCxtAtten(128, 128)
        self.fam = FeatureAggregationModule(128, reduction, window)
        self.freeze_backbone = freeze_backbone

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            print ('Set GCA decoder feature extraction part in eval() mode.')
            self.layer1.eval()
            self.layer2.eval()
            self.gca.eval()

    def forward(self, inputs, extract_feature, x=None, xb=None, xf=None, mask=None):
        if extract_feature:
            x, mid_fea = inputs
            fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
            im = mid_fea['image_fea']
            x = self.layer1(x) + fea5 # N x 256 x 32 x 32
            x = self.layer2(x) + fea4 # N x 128 x 64 x 64
            x, _ = self.gca(im, x, mid_fea['unknown']) # contextual attention
            return x
        else:
            _, mid_fea = inputs
            fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
            x, attb, attf, mask = self.fam(x, xb, xf, mask)
            #x = torch.cat([x, xb, xf], dim=1)
            x = self.layer3(x) + fea3 # N x 64 x 128 x 128
            x = self.layer4(x) + fea2 # N x 32 x 256 x 256
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.leaky_relu(x) + fea1
            x = self.conv2(x)

        alpha = (self.tanh(x) + 1.0) / 2.0

        return alpha, attb, attf, mask#, {'offset_1': mid_fea['offset_1'], 'offset_2': offset}