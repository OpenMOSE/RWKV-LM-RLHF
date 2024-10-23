# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

# Sect4.2 of Linear Transformers Are Secretly Fast Weight Programmers https://arxiv.org/abs/2102.11174
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2_norm_fn
from fla.ops.delta_rule import (chunk_delta_rule, fused_chunk_delta_rule,
                                fused_recurrent_delta_rule)

if TYPE_CHECKING:
    from fla.models.utils import Cache

# @torch.jit.script


def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)

# @torch.jit.script


def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)

# https://github.com/IDSIA/recurrent-fwp/blob/master/algorithmic/layers.py#L86C1-L146C1


class DeltaNet(nn.Module):
    def __init__(
        self,
        d_model: int = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        mode: str = 'chunk',
        chunk_size: int = 64,
        use_beta: bool = True,
        use_gate: bool = False,
        use_output_norm: bool = True,
        use_elu: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        qk_activation: str = 'silu',
        qk_norm: str = 'l2',
        norm_first: bool = False,
        norm_eps: float = 1e-5,
        **kwargs
    ) -> DeltaNet:
        super().__init__()

        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        assert self.qk_activation in ['silu', 'relu', 'elu', 'identity']
        assert self.qk_norm in ['l2', 'sum']

        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.use_gate = use_gate
        self.use_output_norm = use_output_norm
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.norm_first = norm_first
        self.layer_idx = layer_idx

        self.silu = nn.SiLU()

        assert mode in ['chunk', 'fused_chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        if norm_first:
            self.norm = RMSNorm(self.hidden_size, eps=norm_eps)

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.use_beta = use_beta
        self.use_elu = use_elu
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(self.key_dim,
                                             conv_size,
                                             activation='silu' if qk_activation == 'silu' else None)
            self.k_conv1d = ShortConvolution(self.key_dim,
                                             conv_size,
                                             activation='silu' if qk_activation == 'silu' else None)
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        # change to inference mode.
        mode = 'fused_recurrent' if hidden_states.shape[1] < 64 else self.mode

        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        last_state = past_key_values[self.layer_idx] if use_cache else None

        if attention_mask is not None:
            if attention_mask.shape[-1] != hidden_states.shape[-2]:
                attention_mask = attention_mask[:, -1:]

        if self.use_short_conv:
            conv_state_q = last_state[0] if use_cache else None
            conv_state_k = last_state[1] if use_cache else None
            conv_state_v = last_state[2] if use_cache else None
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            q = self.q_proj(hidden_states)
            q = self.q_conv1d(q, attention_mask, conv_state_q)
            k = self.k_conv1d(k, attention_mask, conv_state_k)
            v = self.v_conv1d(v, attention_mask, conv_state_v)
        else:
            q = (self.q_proj(hidden_states))
            k = (self.k_proj(hidden_states))
            v = self.silu(self.v_proj(hidden_states))

        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask.unsqueeze(-1))

        q, k, v = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (q, k, v))

        if self.qk_activation != 'silu':
            if self.qk_activation == 'relu':
                q, k = q.relu(), k.relu()
            elif self.qk_activation == 'elu':
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation == 'identity':
                pass
            else:
                raise NotImplementedError

        if self.qk_norm is not None:
            if self.qk_norm == 'l2':
                q = l2_norm_fn(q)
                k = l2_norm_fn(k)
            elif self.qk_norm == 'sum':
                q = sum_norm(q).to(v)
                k = sum_norm(k).to(v)

        if self.use_beta:
            beta = rearrange(self.b_proj(hidden_states), 'b l h -> b h l').sigmoid()
        else:
            beta = q.new_ones(q.shape[0], q.shape[1], q.shape[2])
        state = past_key_values[self.layer_idx][-1] if use_cache else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_delta_rule(
                q=q, k=k, v=v, beta=beta, initial_state=state, output_final_state=use_cache)
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_delta_rule(
                q=q, k=k, v=v, beta=beta, BT=self.chunk_size, initial_state=state, output_final_state=use_cache)
        elif mode == 'chunk':
            o, recurrent_state = chunk_delta_rule(q=q, k=k, v=v, beta=beta, BT=self.chunk_size,
                                                  initial_state=state, output_final_state=use_cache)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            if self.use_short_conv:
                state = (conv_state_q, conv_state_k, conv_state_v, recurrent_state)
            else:
                state = (recurrent_state,)
            past_key_values.update(state, self.layer_idx)

        o = rearrange(o, 'b h l d -> b l h d')
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), 'b l (h d) -> b l h d', h=self.num_heads)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b l h d -> b l (h d)')
        o = self.o_proj(o)

        return o, None, past_key_values

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = tuple()
        if self.use_short_conv:
            # for q/k/v each
            state += (param.new_zeros(batch_size, self.key_dim, self.conv_size),
                      param.new_zeros(batch_size, self.key_dim, self.conv_size),
                      param.new_zeros(batch_size, self.value_dim, self.conv_size))
        state += (param.new_zeros(batch_size, self.num_heads, self.head_qk_dim, self.head_v_dim),)
        return state
