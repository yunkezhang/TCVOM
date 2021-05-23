import torch
import torch.nn as nn

from . import encoders, decoders


class Generator(nn.Module):
    def __init__(self, encoder, decoder, alpha_only):

        super(Generator, self).__init__()
        self.alpha_only = alpha_only
        if encoder not in encoders.__all__:
            raise NotImplementedError("Unknown Encoder {}".format(encoder))
        self.encoder = encoders.__dict__[encoder]()

        if decoder is None:
            self.decoder = None
        else:
            if decoder not in decoders.__all__:
                raise NotImplementedError("Unknown Decoder {}".format(decoder))
            self.decoder = decoders.__dict__[decoder]()

    def forward(self, inp, **kwargs):
        embedding, mid_fea = self.encoder(inp)
        if self.decoder:
            alpha, info_dict = self.decoder(embedding, mid_fea)
            if self.alpha_only:
                return alpha
            else:
                return alpha, info_dict, embedding
        else:
            return embedding


def GCA(encoder='resnet_gca_encoder_29', decoder='res_gca_decoder_22', alpha_only=True):
    generator = Generator(encoder=encoder, decoder=decoder, alpha_only=alpha_only)
    return generator