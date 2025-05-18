# -*- coding: utf-8 -*-

from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig


class GatedDeltaNetConfig(PretrainedConfig):
    model_type = 'gated_deltanet'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        attn_mode: str = "chunk",
        hidden_size: int = 2048,
        expand_v: int = 2,
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        head_dim: int = 256,
        num_heads: int = 6,
        max_position_embeddings: int = 2048,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
        num_hidden_layers: int = 21,
        norm_first: bool = False,
        norm_eps: float = 1e-6,
        attn: Optional[Dict] = None,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        fuse_cross_entropy: bool = True,
        vocab_size: int = 32000,
        **kwargs
    ):
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_first = norm_first
        self.norm_eps = norm_eps
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_cross_entropy = fuse_cross_entropy
        self.vocab_size = vocab_size

        if attn is not None:
            if not isinstance(attn, Dict):
                raise ValueError("attn must be a dictionary")
            if 'layers' not in attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if 'num_heads' not in attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            attn['num_kv_heads'] = attn.get('num_kv_heads', attn['num_heads'])
            attn['window_size'] = attn.get('window_size', None)
            attn['rope_theta'] = attn.get('rope_theta', 10000.)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
