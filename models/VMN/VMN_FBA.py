import torch

from models.VMN.VMN_model import FeatureAggregationModule
from models.FBA.models import fba_decoder, fba_fusion

class vmn_fba_decoder(fba_decoder):
    def __init__(self, reduction, window, freeze_backbone=False, batch_norm=False):
        super(vmn_fba_decoder, self).__init__(batch_norm=batch_norm)
        self.fam = FeatureAggregationModule(256, reduction, window)
        self.freeze_backbone = freeze_backbone

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            print ('Set FBA decoder feature extraction part in eval() mode.')
            self.conv_up1.eval()

    def forward(self, inputs, extract_feature, x=None, xb=None, xf=None, mask=None):
        conv_out, indices, img, two_chan_trimap = inputs
        if extract_feature:
            conv5 = conv_out[-1]

            input_size = conv5.size()
            ppm_out = [conv5]
            for pool_scale in self.ppm:
                ppm_out.append(torch.nn.functional.interpolate(
                    pool_scale(conv5),
                    (input_size[2], input_size[3]),
                    mode='bilinear', align_corners=False))
            ppm_out = torch.cat(ppm_out, 1)
            x = self.conv_up1(ppm_out)      # OS=8
            return x
        else:
            x, attb, attf, mask = self.fam(x, xb, xf, mask)
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

            x = torch.cat((x, conv_out[-4]), 1)

            x = self.conv_up2(x)
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

            x = torch.cat((x, conv_out[-5]), 1)
            x = self.conv_up3(x)

            x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = torch.cat((x, conv_out[-6][:, :3], img, two_chan_trimap), 1)

            output = self.conv_up4(x)

            alpha = torch.clamp(output[:, :1], 0, 1)
            F = torch.sigmoid(output[:, 1:4])
            B = torch.sigmoid(output[:, 4:7])

            # FBA Fusion
            alpha, F, B = fba_fusion(alpha, img, F, B)

            output = torch.cat((alpha, F, B), 1)

            return output, attb, attf, mask
