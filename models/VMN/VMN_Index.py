import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Index.net import IndexMattingDecoder
from models.VMN.VMN_model import FeatureAggregationModule

class IndexMattingDecoder_VMN(IndexMattingDecoder):
    def __init__(self, reduction, window, freeze_backbone=False):
        super(IndexMattingDecoder_VMN, self).__init__()
        self.fam = FeatureAggregationModule(32, reduction, window)
        self.freeze_backbone = freeze_backbone

    def forward(self, inputs, extract_feature, x=None, xb=None, xf=None, mask=None):
        l, l6, idx6_de, l5, l4, idx4_de, l3, idx3_de, l2, idx2_de, l1, l0, idx0_de = inputs
        if extract_feature:
            # decode                                # out
            l = self.decoder_layer6(l, l6, idx6_de) # OS=16
            l = self.decoder_layer5(l, l5)
            l = self.decoder_layer4(l, l4, idx4_de) # OS=8
            return l
        else:
            x, attb, attf, mask = self.fam(x, xb, xf, mask)
            l = self.decoder_layer3(x, l3, idx3_de) # OS=4
            l = self.decoder_layer2(l, l2, idx2_de) # OS=2
            l = self.decoder_layer1(l, l1)
            l = self.decoder_layer0(l, l0, idx0_de) # OS=1

            l = self.pred(l)
            return l, attb, attf, mask