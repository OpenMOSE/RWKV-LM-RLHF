# -*- coding: utf-8 -*-

from fla.layers import (
                        ReBasedLinearAttention, RWKV6Attention, RWKV7Attention)
from fla.models import ( RWKV6ForCausalLM,
                        RWKV6Model, RWKV7ForCausalLM, RWKV7Model,
                        )

__all__ = [
   
    'RWKV6Attention',
    'RWKV7Attention',
    'RWKV6ForCausalLM',
    'RWKV6Model',
    'RWKV7ForCausalLM',
    'RWKV7Model',
]

__version__ = '0.1'
