# -*- coding: utf-8 -*-

#from fla.modules.convolution import (ImplicitLongConvolution, LongConvolution,
#                                     ShortConvolution)
from rwkvengine.fla.modules.fused_cross_entropy import FusedCrossEntropyLoss
#from fla.modules.fused_norm_gate import (FusedLayerNormSwishGate,
#                                         FusedLayerNormSwishGateLinear,
#                                         FusedRMSNormSwishGate,
#                                         FusedRMSNormSwishGateLinear)
from rwkvengine.fla.modules.layernorm import (GroupNorm, GroupNormLinear, LayerNorm,
                                   LayerNormLinear, RMSNorm, RMSNormLinear)
#from fla.modules.rotary import RotaryEmbedding

__all__ = [
    'ImplicitLongConvolution', 'LongConvolution', 'ShortConvolution',
    'FusedCrossEntropyLoss',
    'GroupNorm', 'GroupNormLinear', 'LayerNorm', 'LayerNormLinear', 'RMSNorm', 'RMSNormLinear',
    'FusedLayerNormSwishGate', 'FusedLayerNormSwishGateLinear', 'FusedRMSNormSwishGate', 'FusedRMSNormSwishGateLinear',
    'RotaryEmbedding'
]
