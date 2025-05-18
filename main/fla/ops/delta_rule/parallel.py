

# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Tuple

import torch
import triton
import triton.language as tl
from einops import rearrange

from fla.ops.delta_rule.wy_fast import fwd_prepare_T
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=["BT", "K", "V"],
)
@triton.jit
def chunk_transform_qk_fwd_kernel(
    q,
    k,
    v,
    beta,
    o,
    A,
    q_new,
    k_new,
    A_local,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BT: tl.constexpr,
    OUTPUT_ATTENTIONS: tl.constexpr,
    # SAVE_ATTENTION: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, 0), (BT, BV), (1, 0))
    b_q = (tl.load(p_q, boundary_check=(0, 1)) * scale).to(p_q.dtype.element_ty)
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))

    p_T = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_T = tl.load(p_T, boundary_check=(0, 1))

    o_i = tl.arange(0, BT)
    m_t = o_i[:, None] >= o_i[None, :]
    b_qk = tl.where(m_t, tl.dot(b_q, tl.trans(b_k), allow_tf32=False), 0).to(b_q.dtype)
    m_t = o_i[:, None] > o_i[None, :]
    b_kk = tl.where(m_t, tl.dot(b_k, tl.trans(b_k), allow_tf32=False), 0).to(b_k.dtype)

    p_beta = tl.make_block_ptr(beta + i_bh * T, (T, ), (1, ), (i_t * BT, ), (BT, ), (0, ))
    b_beta = tl.load(p_beta, boundary_check=(0, ))
    b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)

    b_qkT = tl.dot(b_qk, b_T, allow_tf32=False).to(b_k.dtype)

    if OUTPUT_ATTENTIONS:
        p_a = tl.make_block_ptr(A_local + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        tl.store(p_a, b_qkT.to(p_a.dtype.element_ty), boundary_check=(0, 1))

    b_kkT = tl.dot(b_kk, b_T, allow_tf32=False).to(b_k.dtype)
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, 0), (BT, BV), (1, 0))
    tl.store(p_o, tl.dot(b_qkT, b_v).to(p_o.dtype.element_ty), boundary_check=(0, 1))

    p_q_new = tl.make_block_ptr(q_new + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_q_new, (b_q - tl.dot(b_qkT, b_k_beta, allow_tf32=False)).to(p_q_new.dtype.element_ty), boundary_check=(0, 1))

    p_k_new = tl.make_block_ptr(k_new + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    tl.store(p_k_new, (b_k - tl.dot(tl.trans(b_kkT), b_k_beta, allow_tf32=False)
                       ).to(p_k_new.dtype.element_ty), boundary_check=(0, 1))


def chunk_transform_qk_fwd_fn(q, k, v, beta, A, scale, BT, output_attentions):
    B, H, T, K = k.shape
    q_new = torch.empty_like(q)
    k_new = torch.empty_like(k)
    o = torch.empty_like(v)
    grid = (triton.cdiv(T, BT), B*H)
    V = v.shape[-1]
    A_local = torch.empty_like(A) if output_attentions else None
    chunk_transform_qk_fwd_kernel[grid](
        q, k, v, beta, o, A, q_new, k_new, A_local,
        q.stride(1), q.stride(2), q.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        scale=scale,
        T=T,
        K=K,
        V=V,
        BT=BT,
        BK=triton.next_power_of_2(K),
        BV=triton.next_power_of_2(V),
        OUTPUT_ATTENTIONS=output_attentions
    )
    return q_new, k_new, o, A_local


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
    ],
    key=["BT"],
)
@triton.jit
def save_intra_chunk_attn(
    A,
    A_local,
    T: tl.constexpr,
    BT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_A = tl.make_block_ptr(A + i_bh * T * T, (T, T), (T, 1), (i_t * BT, i_t * BT), (BT, BT), (1, 0))
    p_A_local = tl.make_block_ptr(A_local + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A_local = tl.load(p_A_local, boundary_check=(0, 1))
    tl.store(p_A, b_A_local.to(p_A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'OUTPUT_ATTENTIONS': lambda args: args['attn'] is not None
})
@triton.jit
def parallel_delta_rule_fwd_kernel(
    q,
    k,
    k2,  # original k
    v,
    beta,
    o,
    o_new,
    attn,
    s_k_h,
    s_k_t,
    s_v_h,
    s_v_t,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    OUTPUT_ATTENTIONS: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, 0), (BT, BK), (1, 0))

    # the Q block is kept in the shared memory throughout the whole kernel
    # [BT, BK]
    b_q = tl.zeros([BT, BK], dtype=tl.float32)
    b_q += tl.load(p_q, boundary_check=(0, 1))

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    b_o += tl.load(p_o, boundary_check=(0, 1))

    # As opposed to Flashattention, this kernel requires scanning the KV blocks from right to left
    # Q block and K block have overlap.
    # masks required
    for offset in range((i_t + 1) * BT - 2 * BS, i_t * BT - BS, -BS):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (1, s_k_t), (0, offset), (BK, BS), (0, 1))
        p_k2 = tl.make_block_ptr(k2 + i_bh * s_k_h, (T, K), (s_k_t, 1), (offset, 0), (BS, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (offset, 0), (BS, BV), (1, 0))
        p_beta = tl.make_block_ptr(beta + i_bh * T, (T, ), (1, ), (offset, ), (BS, ), (0,))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BS]
        b_beta = tl.load(p_beta, boundary_check=(0,))
        # [BT, BS]
        m_s = tl.arange(0, BT) >= (offset - i_t*BT + BS)
        b_s = tl.dot(b_q.to(b_k.dtype), b_k, allow_tf32=False)
        b_s = tl.where(m_s[:, None], b_s, 0)

        b_o += tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        b_k2 = (tl.load(p_k2, boundary_check=(0, 1)) * b_beta[:, None]).to(b_v.dtype)
        b_q -= tl.dot(b_s.to(b_v.dtype), b_k2, allow_tf32=False)

        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(attn + i_bh * T * T, (T, T), (T, 1), (i_t * BT, offset), (BT, BS), (1, 0))
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))

    # Q block and K block have no overlap
    # no need for mask, thereby saving flops
    for offset in range(i_t * BT - BS, -BS, -BS):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (1, s_k_t), (0, offset), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (offset, 0), (BS, BV), (1, 0))
        p_beta = tl.make_block_ptr(beta + i_bh * T, (T, ), (1, ), (offset, ), (BS, ), (0,))
        p_k2 = tl.make_block_ptr(k2 + i_bh * s_k_h, (T, K), (s_k_t, 1), (offset, 0), (BS, BK), (1, 0))

        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BS]
        b_beta = tl.load(p_beta, boundary_check=(0,))
        # [BT, BS]
        b_s = (tl.dot(b_q.to(b_k.dtype), b_k, allow_tf32=False))
        # [BT, BV]
        b_o += tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        b_k2 = (tl.load(p_k2, boundary_check=(0, 1)) * b_beta[:, None]).to(b_v.dtype)
        b_q -= tl.dot(b_s.to(b_v.dtype), b_k2, allow_tf32=False).to(b_q.dtype)

        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(attn + i_bh * T * T, (T, T), (T, 1), (i_t * BT, offset), (BT, BS), (1, 0))
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))

    p_o_new = tl.make_block_ptr(o_new + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t*BT, 0), (BT, BV), (1, 0))
    tl.store(p_o_new, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


class ParallelDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, beta, scale, output_attentions):
        B, H, T, K, V = *k.shape, v.shape[-1]
        assert q.shape[-1] <= 128, 'The maximum supported sequence length is 128.'
        BT, BS = 128, 32
        BK = triton.next_power_of_2(k.shape[-1])
        BV = triton.next_power_of_2(v.shape[-1])
        assert BT % BS == 0

        A = fwd_prepare_T(k, beta, BS)
        attn = q.new_zeros(B, H, T, T) if output_attentions else None
        q_new, k_new, o, A_local = chunk_transform_qk_fwd_fn(q, k, v, beta, A, scale, BS, output_attentions)

        num_stages = 3 if K <= 64 else 2
        num_warps = 4
        grid = (triton.cdiv(T, BT), B * H)
        o_new = torch.empty_like(o)

        parallel_delta_rule_fwd_kernel[grid](
            q=q_new,
            k=k_new,
            k2=k,
            v=v,
            beta=beta,
            o=o,
            o_new=o_new,
            attn=attn,
            s_k_h=k.stride(1),
            s_k_t=k.stride(2),
            s_v_h=v.stride(1),
            s_v_t=v.stride(2),
            T=T,
            K=K,
            V=V,
            BT=BT,
            BS=BS,
            BK=BK,
            BV=BV,
            num_stages=num_stages,
            num_warps=num_warps
        )

        if output_attentions:
            grid = (triton.cdiv(T, BS), B * H)
            save_intra_chunk_attn[grid](
                A=attn, A_local=A_local, T=T, BT=BS
            )
        return o_new.to(q.dtype), attn

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, d_attn=None):
        raise NotImplementedError('Backward pass is not implemented. Stay tuned!')


def parallel_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    output_attentions: bool = False,
    head_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        beta (torch.Tensor):
            betas of shape `[B, H, T]` if `head_first=True` else `[B, T, H]`.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        output_attentions (bool):
            Whether to output the materialized attention scores of shape [B, H, T, T]. Default: `False`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format.
            Default: `True`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        attn (torch.Tensor):
            Attention scores of shape `[B, H, T, T]` if `output_attentions=True` else `None`.
    """
    if not head_first:
        q, k, v, beta = map(lambda x: x.transpose(1, 2), (q, k, v, beta))
    o, attn = ParallelDeltaRuleFunction.apply(q, k, v, beta, scale, output_attentions)
    if not head_first:
        o = o.transpose(1, 2)
    return o, attn


def naive_delta_rule_parallel(q, k, v, beta, BM=128, BN=32):
    b, h, l, d_k = q.shape
    q = q * (d_k ** -0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    # compute (I - tri(diag(beta) KK^T))^{-1}
    q, k, v, k_beta = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BN), [q, k, v, k_beta])
    mask = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=q.device), diagonal=0)
    T = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, BN):
        T[..., i, :i] = T[..., i, :i].clone() + (T[..., i, :, None].clone() * T[..., :, :i].clone()).sum(-2)
    T = T + torch.eye(BN, dtype=q.dtype, device=q.device)

    mask2 = torch.triu(torch.ones(BN, BN, dtype=torch.bool, device=q.device), diagonal=1)
    A_local = (q @ k.transpose(-1, -2)).masked_fill(mask2, 0) @ T
    o_intra = A_local @ v

    # apply cumprod transition matrices on k to the last position within the chunk
    k = k - ((k @ k.transpose(-1, -2)).masked_fill(mask, 0) @ T).transpose(-1, -2) @ k_beta
    # apply cumprod transition matrices on q to the first position within the chunk
    q = q - A_local @ k_beta
    o_intra = A_local @ v

    A = torch.zeros(b, h, l, l, device=q.device)

    q, k, v, k_beta, o_intra = map(lambda x: rearrange(x, 'b h n c d -> b h (n c) d'), [q, k, v, k_beta, o_intra])
    o = torch.empty_like(v)
    for i in range(0, l, BM):
        q_i = q[:, :, i:i+BM]
        o_i = o_intra[:, :, i:i+BM]
        # intra block
        for j in range(i + BM - 2 * BN, i-BN, -BN):
            k_j = k[:, :, j:j+BN]
            A_ij = q_i @ k_j.transpose(-1, -2)
            mask = torch.arange(i, i+BM) >= (j + BN)
            A_ij = A_ij.masked_fill_(~mask[:, None].to(A_ij.device), 0)
            A[:, :, i:i+BM, j:j+BN] = A_ij
            q_i = q_i - A_ij @ k_beta[:, :, j:j+BN]
            o_i += A_ij @ v[:, :, j:j+BN]
        # inter block
        for j in range(i - BN, -BN, -BN):
            k_j = k[:, :, j:j+BN]
            A_ij = q_i @ k_j.transpose(-1, -2)
            A[:, :, i:i+BM, j:j+BN] = A_ij
            q_i = q_i - A_ij @ k_beta[:, :, j:j+BN]
            o_i += A_ij @ v[:, :, j:j+BN]
        o[:, :, i:i+BM] = o_i

    for i in range(0, l//BN):
        A[:, :, i*BN:i*BN+BN, i*BN:i*BN+BN] = A_local[:, :, i]

    return o, A


if __name__ == "__main__":
    B, H, T, K, V = 2, 4, 512, 64, 64
    torch.set_default_dtype(torch.bfloat16)

    q = torch.randn[B, H, T, K].cuda()
    k = torch.nn.functional.normalize(torch.randn[B, H, T, K].cuda(), p=2, dim=-1)
    v = torch.randn[B, H, T, V].cuda()
    beta = torch.ones(B, H, T).cuda()

    output_attentions = True
    ref_o, ref_attn = naive_delta_rule_parallel(q.clone(), k.clone(), v.clone(), beta.clone())
    o, attn = parallel_delta_rule(q.clone(), k.clone(), v.clone(), beta.clone(), K**-0.5, output_attentions)
    print((ref_o-o).abs().max())
    print((ref_attn-attn).abs().max())
