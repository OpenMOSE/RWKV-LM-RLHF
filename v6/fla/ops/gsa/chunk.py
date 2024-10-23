# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from einops import reduce

from fla.ops.common.chunk_h import chunk_bwd_dh_fn, chunk_fwd_h_fn
from fla.ops.utils import (chunk_global_reversed_cumsum, chunk_local_cumsum,
                           softmax_bwd_kernel, softmax_fwd_kernel)
from fla.utils import contiguous


@triton.jit
def chunk_gsa_fwd_kernel_intra_K(
    v,
    g,
    o,
    A,
    s_v_h,
    s_v_t,
    T: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    NG: tl.constexpr
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    i_t, i_i = i_c // NC, i_c % NC
    o_v = i_v * BV + tl.arange(0, BV)

    p_g = tl.make_block_ptr(g + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * s_v_h + (i_t * BT + i_i * BC) * V + o_v, BV), BV)
    # [BV,]
    b_gn = tl.load(p_gn, mask=(o_v < V), other=0)
    # [BC, BV]
    b_o = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(0, i_i):
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        p_v = tl.make_block_ptr(v + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        p_gv = tl.make_block_ptr(g + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        # [BC, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_gv = tl.load(p_gv, boundary_check=(0, 1))
        b_vg = (b_v * tl.exp(b_gn[None, :] - b_gv)).to(b_v.dtype)
        # [BC, BC]
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_o += tl.dot(b_A, b_vg)
    # [BC, BV]
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_o *= tl.exp(b_g - b_gn[None, :])

    o_i = tl.arange(0, BC)
    o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    for j in range(0, BC):
        p_v = tl.max_contiguous(tl.multiple_of(v + i_bg * s_v_h + (i_t * BT + i_i * BC + j) * V + o_v, BV), BV)
        p_gv = tl.max_contiguous(tl.multiple_of(g + i_bg * s_v_h + (i_t * BT + i_i * BC + j) * V + o_v, BV), BV)
        # [BC,]
        b_A = tl.load(A + o_A + j, mask=m_A, other=0)
        # [BV,]
        b_v = tl.load(p_v, mask=(o_v < V), other=0).to(tl.float32)
        b_gv = tl.load(p_gv, mask=(o_v < V), other=0).to(tl.float32)
        # [BC, BV]
        b_vg = b_v[None, :] * tl.exp(b_g - b_gv[None, :])
        # avoid 0 * inf = inf
        b_o += tl.where(o_i[:, None] >= j, b_A[:, None] * b_vg, 0.)
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))

    b_o += tl.load(p_o, boundary_check=(0, 1))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gsa_fwd_kernel_K(
    q,
    k,
    h,
    g,
    o,
    A,
    s_k_h,
    s_k_t,
    s_v_h,
    s_v_t,
    s_h_h,
    s_h_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bg * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bg * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BV]
        b_o += tl.dot(b_q, b_h)
        # [BT, BT]
        b_A += tl.dot(b_q, b_k)
    p_g = tl.make_block_ptr(g + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    # [BT, BV]
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_o = b_o * tl.exp(b_g)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [BT, BT]
    b_A = tl.where(m_s, b_A, 0.)
    if i_v == 0:
        tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gsa_fwd_kernel_intra_Vk(
    q,
    k,
    g,
    A,
    s_k_h,
    s_k_t,
    i_k,
    i_c,
    i_bh,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    NG: tl.constexpr
):
    i_bg = i_bh // NG
    i_t, i_i, i_j = i_c // (NC * NC), (i_c % (NC * NC)) // NC, (i_c % (NC * NC)) % NC
    o_k = i_k * BK + tl.arange(0, BK)

    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))

    b_A = tl.zeros([BC, BC], tl.float32)
    if i_i > i_j:
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bg * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g + i_bg * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)

        # [BK,]
        b_gn = tl.load(p_gn, mask=(o_k < K), other=0.)
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = (b_q * tl.exp(b_g - b_gn[None, :]) * scale).to(b_q.dtype)
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = (b_k * tl.exp(b_gn[:, None] - b_gk)).to(b_k.dtype)
        # [BC, BC]
        b_A = tl.dot(b_qg, b_kg)
        if i_k != 0:
            b_A += tl.load(p_A, boundary_check=(0, 1))
        tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))
    elif i_i == i_j:
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.max_contiguous(tl.multiple_of(k + i_bg * s_k_h + (i_t * BT + i_j * BC) * K + o_k, BK), BK)
        p_gk = tl.max_contiguous(tl.multiple_of(g + i_bg * s_k_h + (i_t * BT + i_j * BC) * K + o_k, BK), BK)
        m_k = o_k < K

        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))

        o_i = tl.arange(0, BC)
        # [BC, BC]
        m_A = o_i[:, None] >= o_i[None, :]
        for j in range(0, BC):
            # [BK,]
            b_k = tl.load(p_k, mask=(m_k & (i_t * BT + i_j * BC + j) < T), other=0.).to(tl.float32)
            b_gk = tl.load(p_gk, mask=(m_k & (i_t * BT + i_j * BC + j) < T), other=0.).to(tl.float32)
            # [BC,]
            b_Aj = tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]) * scale, 1)
            b_A = tl.where((o_i == j)[None, :], b_Aj[:, None], b_A)

            p_k += K
            p_gk += K
        b_A = tl.where(m_A, b_A, 0.)
        if i_k != 0:
            b_A += tl.load(p_A, boundary_check=(0, 1))
        tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))
    else:
        # set the upper triangular part to 0
        if i_k == 0:
            tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gsa_fwd_kernel_intra_V(
    q,
    k,
    g,
    A,
    s_k_h,
    s_k_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    NK: tl.constexpr,
    NG: tl.constexpr
):
    i_c, i_bh = tl.program_id(0), tl.program_id(1)

    for i_k in range(0, NK):
        chunk_gsa_fwd_kernel_intra_Vk(
            q,
            k,
            g,
            A,
            s_k_h,
            s_k_t,
            i_k,
            i_c,
            i_bh,
            scale,
            T,
            K,
            BT,
            BC,
            BK,
            NC,
            NG,
        )


@triton.jit
def chunk_gsa_fwd_kernel_V(
    q,
    v,
    g,
    h,
    o,
    A,
    s_k_h,
    s_k_t,
    s_v_h,
    s_v_t,
    s_h_h,
    s_h_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bg * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        # [BT, BK]
        b_qg = (b_q * tl.exp(b_g)).to(b_q.dtype)
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # works but dkw, owing to divine benevolence
        # [BT, BV]
        if i_k >= 0:
            b_o += tl.dot(b_qg, b_h)
    p_v = tl.make_block_ptr(v + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_o += tl.dot(b_A, b_v)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gsa_bwd_kernel_V(
    k,
    v,
    h,
    g,
    A,
    do,
    dh,
    dq,
    dk,
    dv,
    dA,
    s_k_h,
    s_k_t,
    s_v_h,
    s_v_t,
    s_h_h,
    s_h_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    n_bh = tl.num_programs(2)
    o_t = min(i_t * BT + BT, T)
    o_k = i_k * BK + tl.arange(0, BK)

    p_k = tl.make_block_ptr(k + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_gk = tl.make_block_ptr(g + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * s_k_h + (o_t - 1) * K + o_k, BK), BK)
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (BT, T), (1, BT), (0, i_t * BT), (BT, BT), (0, 1))
    m_k = o_k < K

    # [BK,]
    # [BT, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_gn = tl.exp(tl.load(p_gn, mask=m_k, other=0)[None, :] - b_gk)
    b_k = (b_k * b_gn).to(b_k.dtype)
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bg * s_h_h + i_t * V * K, (V, K), (1, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K*V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k*n_bh+i_bh) * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BK, BV]
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dh = b_dh.to(b_k.dtype)

        # [BT, BV]
        b_dv = tl.dot(b_k, b_dh)
        if i_k == 0:
            b_dv += tl.dot(b_A, b_do)
        b_do = (b_do * scale).to(b_do.dtype)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        # [BT, BT]
        b_dA += tl.dot(b_do, tl.trans(b_v))
        # [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        # [BT, BK]
        b_dk += tl.dot(b_v, tl.trans(b_dh))
    b_dq = b_dq * tl.exp(b_gk)
    b_dk = b_dk * b_gn

    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT, ), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    # [BT, BT]
    b_dA = tl.where(m_s, b_dA, 0.).to(b_k.dtype)
    if i_k == 0:
        tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gsa_bwd_kernel_intra_V(
    q,
    k,
    g,
    dA,
    dq,
    dk,
    dg,
    s_k_h,
    s_k_t,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    NG: tl.constexpr,
    OVERWRITE: tl.constexpr
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    i_t, i_i = i_c // NC, i_c % NC
    o_k = i_k * BK + tl.arange(0, BK)

    p_g = tl.make_block_ptr(g + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
    m_k = o_k < K
    # [BK,]
    b_gn = tl.load(p_gn, mask=m_k, other=0.)
    # [BC, BK]
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    for i_j in range(0, i_i):
        p_k = tl.make_block_ptr(k + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_gk = tl.make_block_ptr(g + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        # [BC, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = (b_k * tl.exp(b_gn[None, :] - b_gk)).to(b_k.dtype)
        # [BC, BC]
        b_dA = tl.load(p_dA, boundary_check=(0, 1))
        # [BC, BK]
        b_dq += tl.dot(b_dA, b_kg)
    b_dq *= tl.exp(b_g - b_gn[None, :])

    p_kj = tl.max_contiguous(tl.multiple_of(k + i_bg * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
    p_gkj = tl.max_contiguous(tl.multiple_of(g + i_bg * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
    m_k = o_k < K

    o_i = tl.arange(0, BC)
    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    m_dA = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    for j in range(0, BC):
        # [BC,]
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        # [BK,]
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] >= j
        # [BC, BK]
        b_dq += tl.where(m_i, b_dA[:, None] * b_kj[None, :] * tl.exp(b_g - b_gkj[None, :]), 0.)

        p_kj += K
        p_gkj += K
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))

    b_dq = b_dq + tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()
    p_k = tl.make_block_ptr(k + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_gk = tl.make_block_ptr(g + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * s_k_h + (i_t * BT + i_i * BC + BC - 1) * K + o_k, BK), BK)
    # [BK,]
    b_gn = tl.load(p_gn, mask=m_k, other=0)
    # [BC, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_j * BC, i_i * BC), (BC, BC), (1, 0))
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = (b_q * tl.exp(b_g - b_gn[None, :])).to(b_q.dtype)
        # [BC, BC]
        b_dA = tl.load(p_dA, boundary_check=(0, 1))
        # [BC, BK]
        b_dk += tl.dot(tl.trans(b_dA), b_qg)
    b_dk *= tl.exp(b_gn[None, :] - b_gk)

    p_qj = tl.max_contiguous(tl.multiple_of(q + i_bh * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
    p_gqj = tl.max_contiguous(tl.multiple_of(g + i_bg * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
    m_k = o_k < K

    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC) * BT + i_i * BC + tl.arange(0, BC)
    for j in range(0, BC):
        # [BC,]
        b_dA = tl.load(dA + o_dA + j * BT, mask=(i_t * BT + i_i * BC + j < T), other=0)
        # [BK,]
        b_qj = tl.load(p_qj, mask=m_k, other=0.).to(tl.float32)
        b_gqj = tl.load(p_gqj, mask=m_k, other=0.).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] <= j
        b_dk += tl.where(m_i, b_dA[:, None] * b_qj[None, :] * tl.exp(b_gqj[None, :] - b_gk), 0.)

        p_qj += K
        p_gqj += K
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_dk = b_dk + tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)
    b_dg = b_q * b_dq - b_k * b_dk
    if not OVERWRITE:
        b_dg = b_dg + tl.load(p_dg, boundary_check=(0, 1)).to(tl.float32)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gsa_bwd_kernel_intra_K(
    v,
    g,
    do,
    dA,
    s_v_h,
    s_v_t,
    scale,
    T: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    NG: tl.constexpr
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i, i_j = i_c // (NC * NC), (i_c % (NC * NC)) // NC, (i_c % (NC * NC)) % NC
    i_bg = i_bh // NG
    n_bh = tl.num_programs(2)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    p_dA = tl.make_block_ptr(dA+(i_bh+i_v*n_bh)*T*BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))

    # [BC, BC]
    b_dA = tl.zeros([BC, BC], dtype=tl.float32)
    if i_i > i_j:
        p_v = tl.make_block_ptr(v + i_bg * s_v_h, (V, T), (1, s_v_t), (i_v * BV, i_t * BT + i_j * BC), (BV, BC), (0, 1))
        p_gv = tl.make_block_ptr(g + i_bg * s_v_h, (V, T), (1, s_v_t), (i_v * BV, i_t * BT + i_j * BC), (BV, BC), (0, 1))
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * s_v_h + (i_t * BT + i_i * BC) * V + o_v, BV), BV)
        p_g = tl.make_block_ptr(g + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        # [BV,]
        b_gn = tl.load(p_gn, mask=m_v, other=0.)
        # [BC, BV]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = (b_do * tl.exp(b_g - b_gn[None, :]) * scale).to(b_do.dtype)
        # [BV, BC]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_gv = tl.load(p_gv, boundary_check=(0, 1))
        b_vg = (b_v * tl.exp(b_gn[:, None] - b_gv)).to(b_v.dtype)
        # [BC, BC]
        b_dA = tl.dot(b_do, b_vg)
    elif i_i == i_j:
        p_g = tl.make_block_ptr(g + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_v = tl.max_contiguous(tl.multiple_of(v + i_bg * s_v_h + (i_t * BT + i_j * BC) * V + o_v, BV), BV)
        p_gv = tl.max_contiguous(tl.multiple_of(g + i_bg * s_v_h + (i_t * BT + i_j * BC) * V + o_v, BV), BV)
        # [BC, BV]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)) * scale
        m_v = o_v < V

        o_i = tl.arange(0, BC)
        # [BC, BC]
        m_dA = o_i[:, None] >= o_i[None, :]
        for j in range(0, BC):
            # [BV,]
            b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
            b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
            # [BC,]
            b_dAj = tl.sum(b_do * b_v[None, :] * tl.exp(b_g - b_gv[None, :]), 1)
            b_dA = tl.where((o_i == j)[None, :], b_dAj[:, None], b_dA)

            p_v += V
            p_gv += V
        b_dA = tl.where(m_dA, b_dA, 0.)
    tl.store(p_dA, b_dA.to(dA.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gsa_bwd_kernel_K(
    q,
    k,
    v,
    h,
    g,
    A,
    do,
    dh,
    dq,
    dk,
    dv,
    dA,
    s_k_h,
    s_k_t,
    s_v_h,
    s_v_t,
    s_h_h,
    s_h_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    n_bh = tl.num_programs(2)

    o_i = tl.arange(0, BT)
    o_t = min(i_t * BT + BT, T)
    m_s = o_i[:, None] >= o_i[None, :]

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_A = tl.make_block_ptr(A + (i_k*n_bh+i_bh) * T * BT, (T, BT, ), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))

    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.dot((b_q * scale).to(b_q.dtype), tl.trans(b_k))
    b_A = tl.where(m_s, b_A, 0.)
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        p_v = tl.make_block_ptr(v + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bg * s_h_h + i_t * K*V, (V, K), (1, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_g = tl.make_block_ptr(g + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * s_v_h + (o_t - 1) * V + o_v, BV), BV)
        m_v = o_v < V

        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K*V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k*n_bh+i_bh) * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        # [BV,]
        b_gn = tl.load(p_gn, mask=m_v, other=0)
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_v = (b_v * tl.exp(b_gn[None, :] - b_g)).to(b_v.dtype)
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_h = b_h.to(b_k.dtype)
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = (b_do * tl.exp(b_g) * scale).to(b_do.dtype)
        # [BK, BV]
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dh = b_dh.to(b_k.dtype)

        # [BT, BK]
        b_dq += tl.dot(b_do, b_h)
        b_dk += tl.dot(b_v, tl.trans(b_dh))
        # [BT, BV]
        b_dv = tl.exp(b_gn[None, :] - b_g) * tl.dot(b_k, b_dh)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT, ), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [BT, BT]
    b_dA = tl.load(p_dA, boundary_check=(0, 1))
    # [BT, BK]
    b_dq += tl.dot(b_dA, b_k)
    b_dk += tl.dot(tl.trans(b_dA).to(b_k.dtype), b_q)

    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_gsa_bwd_kernel_intra_KV(
    v,
    g,
    o,
    A,
    do,
    dv,
    dg,
    s_v_h,
    s_v_t,
    T: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    NG: tl.constexpr,
    OVERWRITE: tl.constexpr
):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    i_t, i_i = i_c // NC, i_c % NC
    o_v = i_v * BV + tl.arange(0, BV)

    p_gv = tl.make_block_ptr(g + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_gn = g + i_bg * s_v_h + (i_t * BT + i_i * BC + BC - 1) * V + o_v
    p_gn = tl.max_contiguous(tl.multiple_of(p_gn, BV), BV)
    m_v = o_v < V
    # [BV,]
    b_gn = tl.load(p_gn, mask=m_v, other=0)
    # [BC, BV]
    b_gv = tl.load(p_gv, boundary_check=(0, 1))
    b_dv = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        p_g = tl.make_block_ptr(g + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (BT, T), (1, BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        # [BC, BV]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = (b_do * tl.exp(b_g - b_gn[None, :])).to(b_do.dtype)
        # [BC, BC]
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_dv += tl.dot(b_A, b_do)
    b_dv *= tl.exp(b_gn[None, :] - b_gv)

    o_i = tl.arange(0, BC)
    o_c = i_i * BC + tl.arange(0, BC)

    p_g = tl.max_contiguous(tl.multiple_of(g + i_bg * s_v_h + (i_t * BT + i_i * BC) * V + o_v, BV), BV)
    p_A = tl.max_contiguous(tl.multiple_of(A + i_bh * T * BT + (i_t * BT + i_i * BC) * BT + o_c, BV), BV)
    p_do = tl.max_contiguous(tl.multiple_of(do + i_bh * s_v_h + (i_t * BT + i_i * BC) * V + o_v, BV), BV)

    for j in range(0, BC):
        m_j = i_t * BT + i_i * BC + j < T
        # [BC,]
        b_A = tl.load(p_A, mask=m_j, other=0)
        # [BV,]
        b_g = tl.load(p_g, mask=(m_j & m_v), other=0)
        b_do = tl.load(p_do, mask=(m_j & m_v), other=0)
        # [BC, BV]
        m_i = o_i[:, None] <= j
        b_dv += tl.where(m_i, tl.exp(b_g[None, :] - b_gv) * b_A[:, None] * b_do[None, :], 0.)

        p_g += V
        p_A += BT
        p_do += V
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_v = tl.make_block_ptr(v + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_dg = tl.make_block_ptr(dg + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))

    b_o = tl.load(p_o, boundary_check=(0, 1)).to(tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
    b_dv = b_dv + tl.load(p_dv, boundary_check=(0, 1)).to(tl.float32)
    b_dg = b_o * b_do - b_v * b_dv
    if not OVERWRITE:
        b_dg = b_dg + tl.load(p_dg, boundary_check=(0, 1)).to(tl.float32)
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


def fwd_v(q, k, v, g, B, H, T, K, V, BT, BK, BV, BC, h0=None, output_final_state=False, scale=1.):
    HQ = q.shape[1]
    NT = triton.cdiv(T, BT)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    NC = triton.cdiv(BT, BC)
    NG = HQ // H
    num_warps = 4 if BK == 64 else 2
    num_stages = 1

    h, ht = chunk_fwd_h_fn(
        k=k,
        v=v,
        g=None,
        gk=g,
        gv=None,
        BT=BT,
        h0=h0,
        output_final_state=output_final_state,
        states_in_fp32=False
    )
    A = q.new_empty(B, HQ, T, BT)
    grid = (NT * NC * NC, B * HQ)
    chunk_gsa_fwd_kernel_intra_V[grid](
        q, k, g, A,
        k.stride(1), k.stride(2),
        scale,
        T=T, K=K, BT=BT, BC=BC, BK=BK, NC=NC, NK=NK, NG=NG,
        num_warps=num_warps,
        num_stages=num_stages
    )
    o = v.new_empty(B, HQ, T, V)
    grid = (NV, NT, B * HQ)
    chunk_gsa_fwd_kernel_V[grid](
        q, v, g, h, o, A,
        k.stride(1), k.stride(2),
        v.stride(1), v.stride(2),
        h.stride(1), h.stride(2),
        scale,
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NG=NG,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return o, h, ht, A


def fwd_k(q, k, v, g, B, H, T, K, V, BT, BK, BV, BC, h0=None, output_final_state=False, scale=1.):
    HQ = q.shape[1]
    NT = triton.cdiv(T, BT)
    NV = triton.cdiv(V, BV)
    NC = triton.cdiv(BT, BC)
    NG = HQ // H
    num_warps = 4 if BK == 64 else 2
    num_stages = 1

    h, ht = chunk_fwd_h_fn(
        k=k,
        v=v,
        g=None,
        gk=None,
        gv=g,
        BT=BT,
        h0=h0,
        output_final_state=output_final_state,
        states_in_fp32=False
    )
    o = v.new_empty(B, HQ, T, V)
    A = q.new_empty(B, HQ, T, BT)
    grid = (NV, NT, B * HQ)
    chunk_gsa_fwd_kernel_K[grid](
        q, k, h, g, o, A,
        k.stride(1), k.stride(2),
        v.stride(1), v.stride(2),
        h.stride(1), h.stride(2),
        scale,
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NG=NG,
        num_warps=num_warps,
        num_stages=num_stages
    )
    grid = (NV, NT * NC, B * HQ)
    chunk_gsa_fwd_kernel_intra_K[grid](
        v, g, o, A,
        v.stride(1), v.stride(2),
        T=T, V=V, BT=BT, BC=BC, BV=BV, NC=NC, NG=NG,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return o, h, ht, A


def bwd_v(q, k, v, g, h, h0, A, do, dht, dg, B, H, T, K, V, BT, BK, BV, BC, scale=1.):
    HQ = q.shape[1]
    NT = triton.cdiv(T, BT)
    NK = triton.cdiv(K, BK)
    NC = triton.cdiv(BT, BC)
    NG = HQ // H
    num_warps = 4 if BK == 64 else 2
    num_stages = 1

    overwrite_dg = dg is None
    dh, dh0 = chunk_bwd_dh_fn(
        q=q,
        k=k,
        v=v,
        g=None,
        gk=g,
        gv=None,
        do=do,
        h0=h0,
        dht=dht,
        BT=BT,
        scale=scale,
        states_in_fp32=True
    )
    dq = torch.empty_like(q, dtype=torch.float)
    dk = k.new_empty(B, HQ, T, K, dtype=torch.float)
    dv = v.new_empty(NK, B, HQ, T, V)
    dg = g.new_empty(B, HQ, T, K, dtype=torch.float) if dg is None else dg
    dA = v.new_empty(B, HQ, T, BT)

    grid = (NK, NT, B * HQ)
    chunk_gsa_bwd_kernel_V[grid](
        k, v, h, g, A, do, dh, dq, dk, dv, dA,
        k.stride(1), k.stride(2),
        v.stride(1), v.stride(2),
        h.stride(1), h.stride(2),
        scale,
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NG=NG,
        num_warps=num_warps,
        num_stages=num_stages
    )
    dv = dv.sum(0, dtype=dv.dtype)
    grid = (NK, NT * NC, B * HQ)
    chunk_gsa_bwd_kernel_intra_V[grid](
        q, k, g, dA, dq, dk, dg,
        k.stride(1), k.stride(2),
        T=T, K=K, BT=BT, BC=BC, BK=BK, NC=NC, NG=NG,
        OVERWRITE=overwrite_dg,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return dq, dk, dv, dg, dh0


def bwd_k(q, k, v, g, h, h0, o, do, dht, dg, B, H, T, K, V, BT, BK, BV, BC, scale=1.):
    HQ = q.shape[1]
    NT = triton.cdiv(T, BT)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    NC = triton.cdiv(BT, BC)
    NG = HQ // H
    num_warps = 4 if BK == 64 else 2
    num_stages = 1

    overwrite_dg = dg is None
    dh, dh0 = chunk_bwd_dh_fn(
        q=q,
        k=k,
        v=v,
        g=None,
        gk=None,
        gv=g,
        do=do,
        h0=h0,
        dht=dht,
        BT=BT,
        scale=scale,
        states_in_fp32=True
    )
    dA = q.new_empty(NV, B, HQ, T, BT)
    grid = (NV, NT * NC * NC, B * HQ)
    chunk_gsa_bwd_kernel_intra_K[grid](
        v, g, do, dA,
        v.stride(1), v.stride(2),
        scale,
        T=T, V=V, BT=BT, BC=BC, BV=BV, NC=NC, NG=NG,
        num_warps=num_warps,
        num_stages=num_stages
    )
    dA = dA.sum(0, dtype=dA.dtype)

    A = do.new_empty(NK, B, HQ, T, BT)
    dq = torch.empty_like(q)
    dk = k.new_empty(B, HQ, T, K)
    dv = v.new_empty(NK, B, HQ, T, V)
    dg = g.new_empty(B, HQ, T, V, dtype=torch.float) if dg is None else dg
    grid = (NK, NT, B * HQ)
    chunk_gsa_bwd_kernel_K[grid](
        q, k, v, h, g, A, do, dh, dq, dk, dv, dA,
        q.stride(1), q.stride(2),
        v.stride(1), v.stride(2),
        h.stride(1), h.stride(2),
        scale,
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NG=NG,
        num_warps=num_warps,
        num_stages=num_stages
    )
    A = A.sum(0, dtype=A.dtype)
    dv = dv.sum(0, dtype=dv.dtype)
    grid = (NV, NT * NC, B * HQ)
    chunk_gsa_bwd_kernel_intra_KV[grid](
        v, g, o, A, do, dv, dg,
        v.stride(1), v.stride(2),
        T=T, V=V, BT=BT, BC=BC, BV=BV, NC=NC, NG=NG,
        OVERWRITE=overwrite_dg,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return dq, dk, dv, dg, dh0


class ChunkGSAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, s, g, scale, hk0, hv0, output_final_state, checkpoint_level):
        B, H, T, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
        BT, BC = 64, 16
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))
        BM = min(64, triton.next_power_of_2(M))

        g_org, g = g, chunk_local_cumsum(g, BT)
        ok, hk, hkt, Ak = fwd_k(
            q=q, k=k, v=s, g=g,
            B=B, H=H, T=T, K=K, V=M, BT=BT, BK=BK, BV=BM, BC=BC,
            h0=hk0,
            output_final_state=output_final_state,
            scale=scale
        )

        # equivalent to:
        # p = ok.softmax(-1, torch.float)
        # p is kept in fp32 for safe softmax backward
        p = torch.empty_like(ok, dtype=torch.float)
        def grid(meta): return (triton.cdiv(meta['T'], meta['BT']), p.shape[0] * p.shape[1])
        softmax_fwd_kernel[grid](
            ok, p,
            s.stride(1), s.stride(2), s.stride(3),
            T=T, S=M, BT=BT
        )
        qv = p.to(q.dtype)

        ov, hv, hvt, Av = fwd_v(
            q=qv, k=s, v=v, g=g,
            B=B, H=H, T=T, K=M, V=V, BT=BT, BK=BM, BV=BV, BC=BC,
            h0=hv0,
            output_final_state=output_final_state,
            scale=1.
        )

        if checkpoint_level >= 1:
            del g
            g = g_org
        if checkpoint_level > 1:
            del hk
            del hv
            hk, hv = None, None
        else:
            hk0, hv0 = None, None

        ctx.save_for_backward(q, k, v, s, g, ok, p, hk, hv, Av, hk0, hv0)
        ctx.checkpoint_level = checkpoint_level
        ctx.scale = scale
        ctx.BT = BT
        return ov, hkt, hvt

    @staticmethod
    @contiguous
    def backward(ctx, dov, dhkt=None, dhvt=None):
        q, k, v, s, g, ok, p, hk, hv, Av, hk0, hv0 = ctx.saved_tensors
        qv = p.to(q.dtype)
        B, H, T, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
        BT, BC = ctx.BT, 16
        BK = min(64, triton.next_power_of_2(K))
        BV = min(64, triton.next_power_of_2(V))
        BM = min(64, triton.next_power_of_2(M))

        if ctx.checkpoint_level >= 1:
            g = chunk_local_cumsum(g, BT)

        # rerun the forward pass to get h if checkpoint_level >= 1
        if ctx.checkpoint_level > 1:
            hk, _ = chunk_fwd_h_fn(
                k=k,
                v=s,
                g=None,
                gk=None,
                gv=g,
                BT=BT,
                h0=hk0,
                output_final_state=False,
                states_in_fp32=False
            )
            hv, _ = chunk_fwd_h_fn(
                k=s,
                v=v,
                g=None,
                gk=g,
                gv=None,
                BT=BT,
                h0=hv0,
                output_final_state=False,
                states_in_fp32=False
            )

        dqv, dsv, dv, dg, dhv0 = bwd_v(
            q=qv, k=s, v=v, g=g, h=hv, h0=hv0, A=Av, do=dov, dht=dhvt, dg=None,
            B=B, H=H, T=T, K=M, V=V, BT=BT, BK=BM, BV=BV, BC=BC,
            scale=1.
        )

        # softmax gradient, equivalent to:
        # dok = qv * (dqv - (qv * dqv).sum(-1, True))
        dok = torch.empty_like(ok)
        def grid(meta): return (triton.cdiv(meta['T'], meta['BT']), p.shape[0] * p.shape[1])
        softmax_bwd_kernel[grid](
            p, dqv, dok,
            s.stride(1), s.stride(2), s.stride(3),
            T=T, S=M, BT=BT
        )

        dq, dk, dsk, dg, dhk0 = bwd_k(
            q=q, k=k, v=s, g=g, h=hk, h0=hk0, o=ok, do=dok, dht=dhkt, dg=dg,
            B=B, H=H, T=T, K=K, V=M, BT=BT, BK=BK, BV=BM, BC=BC,
            scale=ctx.scale
        )

        ds = dsv.add_(dsk)
        # reversed cumsum, equivalent to:
        #
        # def reversed_cumsum(x, dim=-1):
        #     c = x.cumsum(dim)
        #     return x + c.index_select(dim, x.new_tensor([c.shape[dim]-1], dtype=torch.long)) - c
        dg = chunk_global_reversed_cumsum(dg).to(s.dtype)
        if q.shape[1] != H:
            dk, dv, ds, dg = map(lambda x: reduce(x, 'b (h g) ... -> b h ...', 'sum', h=H), (dk, dv, ds, dg))
        return dq, dk, dv, ds, dg, None, dhk0, dhv0, None, None


def chunk_gsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[int] = None,
    initial_state: Optional[Tuple[torch.Tensor]] = None,
    output_final_state: Optional[bool] = False,
    checkpoint_level: Optional[int] = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `(B, HQ, T, K)`.
        k (torch.Tensor):
            keys of shape `(B, H, T, K)`. GQA is performed if `H` is not equal to `HQ`.
        v (torch.Tensor):
            values of shape `(B, H, T, V)`.
        g (torch.Tensor):
            Forget gates of shape `(B, H, T, M)` applied to keys.
            If not provided, this function is equivalent to vanilla ABC.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[Tuple[torch.Tensor]]):
            Initial state tuple having tensors of shape `(B, H, K, V)`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state tuple, having tensors of shape `(B, H, K, V)`. Default: `False`.
        checkpoint_level (Optional[int]):
            Checkpointing level; higher values will save more memories and do more recomputations during backward.
            Default: `2`:
            - Level `0`: no memory saved, no recomputation.
            - Level `1`: recompute the fp32 cumulative values during backward.
            - Level `2`: recompute the fp32 cumulative values and forward hidden states during backward.
    """
    assert checkpoint_level in [0, 1, 2]
    if g is None:
        # TODO: this 3 steps took huge amount of time, ought to be optimized
        z = s.float().logcumsumexp(2)
        g = torch.cat((z[:, :, :1], z[:, :, :-1]), 2) - z
        s = torch.exp(s - z).to(k.dtype)
    if scale is None:
        scale = q.shape[-1] ** -0.5

    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state
    ov, *final_state = ChunkGSAFunction.apply(q, k, v, s, g, scale, hk0, hv0, output_final_state, checkpoint_level)
    return ov, final_state
