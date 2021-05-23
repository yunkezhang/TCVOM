from models.GCA import encoders as GCA_encoders
from models.FBA.models import ModelBuilder as FBAModelBuilder
from models.Index.net import IndexMattingEncoder

from models.VMN.VMN_DIM import DIMDecoder, DIMEncoder
from models.VMN.VMN_GCA import ResGuidedCxtAtten_FAM_Dec
from models.VMN.VMN_FBA import vmn_fba_decoder
from models.VMN.VMN_Index import IndexMattingDecoder_VMN
from models.VMN.VMN_model import VMN

def get_VMN_models(arch, agg_window, agg_reduction=1, freeze_backbone=False, **kwargs):
    if arch == 'vmn_gca':
        e = GCA_encoders.__dict__['resnet_gca_encoder_29']()
        dn = ResGuidedCxtAtten_FAM_Dec
    elif arch == 'vmn_dim':
        e = DIMEncoder(4)
        dn = DIMDecoder
    elif arch == 'vmn_fba':
        builder = FBAModelBuilder()
        e = builder.build_encoder(arch='resnet50_GN_WS')
        dn = vmn_fba_decoder
    elif arch == 'vmn_index':
        e = IndexMattingEncoder()
        dn = IndexMattingDecoder_VMN
    else:
        raise ValueError
    d = dn(agg_reduction, agg_window, freeze_backbone=freeze_backbone)
    model = VMN(encoder=e, decoder=d, freeze_backbone=freeze_backbone) 
    return model
