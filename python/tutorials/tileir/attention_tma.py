"""
Fused Attention
===============

This is tileIR backend friendly version of the Triton implementation of the Flash Attention v2 

"""

import math
import inspect
from typing import Optional

import triton
import triton.language as tl
import torch
from triton.tools.tensor_descriptor import TensorDescriptor

INV_LOG_2 = tl.constexpr(1.0 / math.log(2))

def supports_host_descriptor():
    return torch.cuda.get_device_capability()[0] >= 9

def supports_tma(tensor: torch.Tensor):
    # Check if the tensor stride is divisible by 16 bytes
    # Mainly for the non-even sequence length case
    return torch.finfo(tensor.dtype).bits * tensor.stride(-2) // 8 % 16 == 0

def _host_descriptor_pre_hook_bwd(nargs):
    print(nargs, flush=True)
    if "BLOCK_M" in nargs:
        BLOCK_M = nargs["BLOCK_M"]
        BLOCK_D = nargs["BLOCK_D"]
    else:
        BLOCK_M_dkdv = nargs["BLOCK_M_dkdv"]
        BLOCK_N_dkdv = nargs["BLOCK_N_dkdv"]
        BLOCK_M_dq = nargs["BLOCK_M_dq"]
        BLOCK_N_dq = nargs["BLOCK_N_dq"]
        BLOCK_D = nargs["BLOCK_D"]
        BLK_SLICE_FACTOR = nargs["BLK_SLICE_FACTOR"]
    # bwd preprocess kernel
    if "minus_L" in nargs:
        if not isinstance(nargs["dO"], TensorDescriptor):
            return
        nargs["dO"].block_shape = [1, 1, BLOCK_M, BLOCK_D]
        nargs["Out"].block_shape = [1, 1, BLOCK_M, BLOCK_D]
        if nargs["USE_DESC_FOR_CORR"]:
            nargs["minus_L"].block_shape = [1, 1, BLOCK_M]
            nargs["minus_Delta"].block_shape = [1, 1, BLOCK_M]
            nargs["L"].block_shape = [1, 1, BLOCK_M]
    # bwd kernel
    else:
        if not isinstance(nargs["Q_dq"], TensorDescriptor):
            return
        nargs["Q_dq"].block_shape = [1, 1, BLOCK_M_dq, BLOCK_D]
        nargs["Q_dkdv_mask"].block_shape = [
            1,
            1,
            BLOCK_M_dkdv // BLK_SLICE_FACTOR,
            BLOCK_D,
        ]
        nargs["Q_dkdv"].block_shape = [1, 1, BLOCK_M_dkdv, BLOCK_D]
        nargs["K_dkdv"].block_shape = [1, 1, BLOCK_N_dkdv, BLOCK_D]
        nargs["K_dq_mask"].block_shape = [
            1,
            1,
            BLOCK_N_dq // BLK_SLICE_FACTOR,
            BLOCK_D,
        ]
        nargs["K_dq"].block_shape = [1, 1, BLOCK_N_dq, BLOCK_D]
        nargs["V_dkdv"].block_shape = [1, 1, BLOCK_N_dkdv, BLOCK_D]
        nargs["V_dq_mask"].block_shape = [
            1,
            1,
            BLOCK_N_dq // BLK_SLICE_FACTOR,
            BLOCK_D,
        ]
        nargs["V_dq"].block_shape = [1, 1, BLOCK_N_dq, BLOCK_D]
        nargs["dO_dq"].block_shape = [1, 1, BLOCK_M_dq, BLOCK_D]
        nargs["dO_dkdv_mask"].block_shape = [
            1,
            1,
            BLOCK_M_dkdv // BLK_SLICE_FACTOR,
            BLOCK_D,
        ]
        nargs["dO_dkdv"].block_shape = [1, 1, BLOCK_M_dkdv, BLOCK_D]
        nargs["dQ"].block_shape = [1, 1, BLOCK_M_dq, BLOCK_D]
        nargs["dK"].block_shape = [1, 1, BLOCK_N_dkdv, BLOCK_D]
        nargs["dV"].block_shape = [1, 1, BLOCK_N_dkdv, BLOCK_D]
        if nargs["USE_DESC_FOR_CORR"]:
            nargs["L_dq"].block_shape = [1, 1, BLOCK_M_dq]
            nargs["L_dkdv_mask"].block_shape = [
                1,
                1,
                BLOCK_M_dkdv // BLK_SLICE_FACTOR,
            ]
            nargs["L_dkdv"].block_shape = [1, 1, BLOCK_M_dkdv]
            nargs["Delta_dq"].block_shape = [1, 1, BLOCK_M_dq]
            nargs["Delta_dkdv_mask"].block_shape = [
                1,
                1,
                BLOCK_M_dkdv // BLK_SLICE_FACTOR,
            ]
            nargs["Delta_dkdv"].block_shape = [1, 1, BLOCK_M_dkdv]

# for fp8 bwd on sm100, we need to make sure BLOCK_K greater equal to 32
def early_config_prune_bwd_two_loop(configs, named_args, **kwargs):
    if torch.cuda.get_device_capability() == (10, 0):
        dtype = kwargs['dtype']
        if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
            return [
                conf
                for conf in configs
                if conf.kwargs['BLOCK_M_dkdv']
                // conf.kwargs['BLK_SLICE_FACTOR']
                >= 32
                and conf.kwargs['BLOCK_N_dq']
                // conf.kwargs['BLK_SLICE_FACTOR']
                >= 32
            ]
    return configs

def get_configs_bwd(is_preprocess=False):
    if supports_host_descriptor():
        _hook = (
            _host_descriptor_pre_hook_bwd
        )
    else:
        _hook = None
    if is_preprocess:
        return [
            triton.Config(
                {
                    'BLOCK_M': 128,
                    'BLOCK_N': 64,
                    'warp_specialize': False,
                },
                pre_hook=_hook,
            ),
        ]
    if torch.cuda.get_device_capability() == (12, 0):
        return [
            triton.Config(
                {
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "occupancy": 2,
                    "warp_specialize": False,
                },
                pre_hook=_hook,
            ),
        ]
    else:
        return [
            triton.Config(
                {
                    'BLOCK_M_dkdv': 32,
                    'BLOCK_N_dkdv': 128,
                    'BLOCK_M_dq': 128,
                    'BLOCK_N_dq': 32,
                    'BLK_SLICE_FACTOR': block_slice_factor,
                    'warp_specialize': False,
                },
                pre_hook=_hook,
            )
            for block_slice_factor in [1, 2]
        ]
    return configs

@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(
            desc_or_ptr, shape, strides, block_shape
        )

@triton.autotune(
    configs=get_configs_bwd(is_preprocess=True),
    key=["S_qo", "USE_DESC_FOR_CORR"],
)
@triton.jit
def fmha_bwd_preprocess_kernel(
    Out,
    dO,
    L,
    minus_Delta,
    minus_L,
    stride_ob,
    stride_oh,
    stride_om,
    stride_lb,
    stride_lh,
    B,
    H,
    S_qo,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_DESC_FOR_CORR: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    O_desc = _maybe_make_tensor_desc(
        Out,
        shape=[B, H, S_qo, BLOCK_D],
        strides=[stride_ob, stride_oh, stride_om, 1],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    dO_desc = _maybe_make_tensor_desc(
        dO,
        shape=[B, H, S_qo, BLOCK_D],
        strides=[stride_ob, stride_oh, stride_om, 1],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    if USE_DESC_FOR_CORR:
        L_desc = _maybe_make_tensor_desc(
            L,
            shape=[B, H, S_qo],
            strides=[stride_lb, stride_lh, 1],
            block_shape=[1, 1, BLOCK_M],
        )
        minus_Delta_desc = _maybe_make_tensor_desc(
            minus_Delta,
            shape=[B, H, S_qo],
            strides=[stride_lb, stride_lh, 1],
            block_shape=[1, 1, BLOCK_M],
        )
        minus_L_desc = _maybe_make_tensor_desc(
            minus_L,
            shape=[B, H, S_qo],
            strides=[stride_lb, stride_lh, 1],
            block_shape=[1, 1, BLOCK_M],
        )
    offset_b = pid_y // H
    offset_h = pid_y % H
    offset_s = (pid_x * BLOCK_M).to(tl.int32)
    o = O_desc.load([offset_b, offset_h, offset_s, 0]).to(tl.float32)
    do = dO_desc.load([offset_b, offset_h, offset_s, 0]).to(tl.float32)
    offset_s_ptrs = offset_s + tl.reshape(
        tl.arange(0, BLOCK_M), (1, 1, BLOCK_M)
    )
    offset_ptrs = offset_b * stride_lb + offset_h * stride_lh + offset_s_ptrs
    if USE_DESC_FOR_CORR:
        l = L_desc.load([offset_b, offset_h, offset_s])
    else:
        l = tl.load(L + offset_ptrs, mask=offset_s_ptrs < S_qo, other=0.0)

    delta = -tl.sum(o * do, axis=3) * softmax_scale
    ml = -l
    if USE_DESC_FOR_CORR:
        minus_Delta_desc.store([offset_b, offset_h, offset_s], delta)
        minus_L_desc.store([offset_b, offset_h, offset_s], ml)
    else:
        tl.store(minus_Delta + offset_ptrs, delta, mask=offset_s_ptrs < S_qo)
        tl.store(minus_L + offset_ptrs, ml, mask=offset_s_ptrs < S_qo)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,  #
    Q,
    k,
    v,
    sm_scale,  #
    DO,  #
    L,
    Delta,  #
    off_b,
    off_h,
    stride_lb,
    stride_lh,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    # Filled in by the wrapper.
    start_n,
    start_m,
    num_steps,
    dtype,  #
    N_CTX,
    USE_DESC_FOR_CORR: tl.constexpr,
    MASK: tl.constexpr,
):
    softmax_scale_inv_ln2 = sm_scale * tl.constexpr(
        1.0 / math.log(2)
    )  # INV_LOG_2
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    base_offset = off_b * stride_lb + off_h * stride_lh
    for blk_idx in range(num_steps):
        qT = Q.load([off_b, off_h, curr_m, 0]).reshape(BLOCK_M1, HEAD_DIM).T
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        offs_m_3d = tl.reshape(offs_m, (1, 1, BLOCK_M1))
        if USE_DESC_FOR_CORR:
            m = L.load([off_b, off_h, curr_m]).reshape(BLOCK_M1)
        else:
            m = tl.load(
                L + base_offset + offs_m_3d, mask=offs_m_3d < N_CTX, other=0.0
            ).reshape(BLOCK_M1)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT * softmax_scale_inv_ln2 + m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)
        do = DO.load([off_b, off_h, curr_m, 0]).reshape(BLOCK_M1, HEAD_DIM)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(dtype)
        dv += tl.dot(ppT, do)
        if USE_DESC_FOR_CORR:
            Di = Delta.load([off_b, off_h, curr_m]).reshape(BLOCK_M1)
        else:
            Di = tl.load(
                Delta + base_offset + offs_m_3d,
                mask=offs_m_3d < N_CTX,
                other=0.0,
            ).reshape(BLOCK_M1)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT * sm_scale + Di[None, :])
        dsT = dsT.to(dtype)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
    dq,
    q,
    K,
    V,  #
    do,
    m,
    Di,
    sm_scale,
    off_b,
    off_h,
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    # Filled in by the wrapper.
    start_m,
    start_n,
    num_steps,
    dtype,  #
    MASK: tl.constexpr,
):
    softmax_scale_inv_ln2 = sm_scale * tl.constexpr(
        1.0 / math.log(2)
    )  # INV_LOG_2
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = K.load([off_b, off_h, curr_n, 0]).reshape(BLOCK_N2, HEAD_DIM).T
        vT = V.load([off_b, off_h, curr_n, 0]).reshape(BLOCK_N2, HEAD_DIM).T
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk * softmax_scale_inv_ln2 + m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp * sm_scale + Di)
        ds = ds.to(dtype)
        # Compute dQ.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
    return dq


    if isinstance(Q, tl.tensor_descriptor):
        dtype = Q.type.block_type.element_ty
    else:
        dtype = Q.dtype.element_ty

    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    if not IS_CAUSAL:
        start_m = 0
    else:
        start_m = ((pid_x * BLOCK_N) // BLOCK_M) * BLOCK_M


    Q_desc = _maybe_make_tensor_desc(
        Q,
        shape=[B, H, S_qo, BLOCK_D],
        strides=[stride_qb, stride_qh, stride_qm, 1],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    K_desc = _maybe_make_tensor_desc(
        K,
        shape=[B, H, S_kv, BLOCK_D],
        strides=[stride_kb, stride_kh, stride_kn, 1],
        block_shape=[1, 1, BLOCK_N, BLOCK_D],
    )
    V_desc = _maybe_make_tensor_desc(
        V,
        shape=[B, H, S_kv, BLOCK_D],
        strides=[stride_vb, stride_vh, stride_vk, 1],
        block_shape=[1, 1, BLOCK_N, BLOCK_D],
    )
    dO_desc = _maybe_make_tensor_desc(
        dO,
        shape=[B, H, S_qo, BLOCK_D],
        strides=[stride_ob, stride_oh, stride_om, 1],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    dQ_desc = _maybe_make_tensor_desc(
        dQ,
        shape=[B, H, S_qo, BLOCK_D],
        strides=[stride_qb, stride_qh, stride_qm, 1],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )

    dK_desc = _maybe_make_tensor_desc(
        dK,
        shape=[B, H, S_kv, BLOCK_D],
        strides=[stride_kb, stride_kh, stride_kn, 1],
        block_shape=[1, 1, BLOCK_N, BLOCK_D],
    )
    dV_desc = _maybe_make_tensor_desc(
        dV,
        shape=[B, H, S_kv, BLOCK_D],
        strides=[stride_vb, stride_vh, stride_vk, 1],
        block_shape=[1, 1, BLOCK_N, BLOCK_D],
    )
    if USE_DESC_FOR_CORR:
        L_desc = _maybe_make_tensor_desc(
            L,
            shape=[B, H, S_qo],
            strides=[stride_lb, stride_lh, 1],
            block_shape=[1, 1, BLOCK_M],
        )
        Delta_desc = _maybe_make_tensor_desc(
            Delta,
            shape=[B, H, S_qo],
            strides=[stride_lb, stride_lh, 1],
            block_shape=[1, 1, BLOCK_M],
        )

    offset_b = pid_y // H
    offset_h = pid_y % H
    offset_skv = pid_x * BLOCK_N

    offset_s_ptrs = start_m + tl.reshape(tl.arange(0, BLOCK_M), (1, 1, BLOCK_M))
    offset_ptrs = offset_b * stride_lb + offset_h * stride_lh + offset_s_ptrs

    k = K_desc.load([offset_b, offset_h, offset_skv, 0]).reshape(BLOCK_N, BLOCK_D)
    v = V_desc.load([offset_b, offset_h, offset_skv, 0]).reshape(BLOCK_N, BLOCK_D)

    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    softmax_scale_inv_ln2 = softmax_scale * tl.constexpr(1.0 / math.log(2))  # INV_LOG_2

    for curr_m in range(start_m, S_qo, BLOCK_M, warp_specialize):
        curr_m = tl.multiple_of(curr_m, BLOCK_M)
        q = Q_desc.load([offset_b, offset_h, curr_m, 0]).reshape(BLOCK_M, BLOCK_D)
        s_t = tl.dot(k, tl.trans(q))
        do = dO_desc.load([offset_b, offset_h, curr_m, 0]).reshape(BLOCK_M, BLOCK_D)
        dp_t = tl.dot(v, tl.trans(do))
        if USE_DESC_FOR_CORR:
            l = L_desc.load([offset_b, offset_h, curr_m])
        else:
            l = tl.load(L + offset_ptrs, mask=offset_s_ptrs < S_qo, other=0.0)
        if IS_CAUSAL:
            offs_n = offset_skv + tl.arange(0, BLOCK_N)
            offs_m_curr = curr_m + tl.arange(0, BLOCK_M)
            s_t = tl.where(offs_m_curr[None, :] >= offs_n[:, None], s_t, float("-inf"))
        if USE_DESC_FOR_CORR:
            delta = Delta_desc.load([offset_b, offset_h, curr_m])
        else:
            delta = tl.load(Delta + offset_ptrs, mask=offset_s_ptrs < S_qo, other=0.0)
        l_t = tl.view(l, (1, BLOCK_M))
        # replace sub l_t to add l_t to use fma2 instruction
        s_t = softmax_scale_inv_ln2 * s_t + l_t
        p_t = tl.exp2(s_t)
        p_t_f16 = p_t.to(dtype)
        delta_t = tl.view(delta, (1, BLOCK_M))
        dp_t_new = softmax_scale * dp_t + delta_t
        ds_t = (p_t * dp_t_new).to(dtype)
        dk = tl.dot(ds_t, q, dk)
        dv = tl.dot(p_t_f16, do, dv)
        ds = tl.trans(ds_t)
        dq = tl.dot(ds, k)
        dQ_desc.atomic_add(
            [offset_b, offset_h, curr_m, 0],
            dq.reshape(1, 1, BLOCK_M, BLOCK_D),
        )
        if not USE_DESC_FOR_CORR:
            offset_s_ptrs += BLOCK_M
            offset_ptrs += BLOCK_M
    dK_desc.store([offset_b, offset_h, offset_skv, 0], dk.to(dtype).reshape(1, 1, BLOCK_N, BLOCK_D))
    dV_desc.store([offset_b, offset_h, offset_skv, 0], dv.to(dtype).reshape(1, 1, BLOCK_N, BLOCK_D))

@triton.autotune(
    configs=get_configs_bwd(),
    key=["S_qo", "USE_DESC_FOR_CORR", "dtype"],
    prune_configs_by={'early_config_prune': early_config_prune_bwd_two_loop},
)
@triton.jit
def fmha_bwd_two_loop_kernel(
    Q_dq,
    Q_dkdv_mask,
    Q_dkdv,
    K_dkdv,
    K_dq_mask,
    K_dq,
    V_dkdv,
    V_dq_mask,
    V_dq,
    dO_dq,
    dO_dkdv_mask,
    dO_dkdv,
    dQ,
    dK,
    dV,
    L_dq,
    L_dkdv_mask,
    L_dkdv,
    Delta_dq,
    Delta_dkdv_mask,
    Delta_dkdv,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vk,
    stride_ob,
    stride_oh,
    stride_om,
    stride_lb,
    stride_lh,
    B,
    H,
    S_qo,
    S_kv,
    BLOCK_M_dkdv: tl.constexpr,
    BLOCK_N_dkdv: tl.constexpr,
    BLOCK_M_dq: tl.constexpr,
    BLOCK_N_dq: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_DESC_FOR_CORR: tl.constexpr,
    dtype: tl.constexpr,
    warp_specialize: tl.constexpr,
):

    if isinstance(Q_dq, tl.tensor_descriptor):
        dtype = Q_dq.type.block_type.element_ty
    else:
        dtype = Q_dq.dtype.element_ty

    BLOCK_M_dkdv_mask: tl.constexpr = BLOCK_M_dkdv // BLK_SLICE_FACTOR
    BLOCK_N_dq_mask: tl.constexpr = BLOCK_N_dq // BLK_SLICE_FACTOR

    pid = tl.program_id(0)
    bhid = tl.program_id(1)
    off_b = bhid // H
    off_h = bhid % H

    start_n = pid * BLOCK_N_dkdv
    start_m = start_n

    Q_dq_desc = _maybe_make_tensor_desc(
        Q_dq,
        shape=[B, H, S_qo, BLOCK_D],
        strides=[stride_qb, stride_qh, stride_qm, 1],
        block_shape=[1, 1, BLOCK_M_dq, BLOCK_D],
    )
    Q_dkdv_mask_desc = _maybe_make_tensor_desc(
        Q_dkdv_mask,
        shape=[B, H, S_qo, BLOCK_D],
        strides=[stride_qb, stride_qh, stride_qm, 1],
        block_shape=[1, 1, BLOCK_M_dkdv_mask, BLOCK_D],
    )
    Q_dkdv_desc = _maybe_make_tensor_desc(
        Q_dkdv,
        shape=[B, H, S_qo, BLOCK_D],
        strides=[stride_qb, stride_qh, stride_qm, 1],
        block_shape=[1, 1, BLOCK_M_dkdv, BLOCK_D],
    )
    K_dkdv_desc = _maybe_make_tensor_desc(
        K_dkdv,
        shape=[B, H, S_kv, BLOCK_D],
        strides=[stride_kb, stride_kh, stride_kn, 1],
        block_shape=[1, 1, BLOCK_N_dkdv, BLOCK_D],
    )
    K_dq_mask_desc = _maybe_make_tensor_desc(
        K_dq_mask,
        shape=[B, H, S_kv, BLOCK_D],
        strides=[stride_kb, stride_kh, stride_kn, 1],
        block_shape=[1, 1, BLOCK_N_dq_mask, BLOCK_D],
    )
    K_dq_desc = _maybe_make_tensor_desc(
        K_dq,
        shape=[B, H, S_kv, BLOCK_D],
        strides=[stride_kb, stride_kh, stride_kn, 1],
        block_shape=[1, 1, BLOCK_N_dq, BLOCK_D],
    )
    V_dkdv_desc = _maybe_make_tensor_desc(
        V_dkdv,
        shape=[B, H, S_kv, BLOCK_D],
        strides=[stride_vb, stride_vh, stride_vk, 1],
        block_shape=[1, 1, BLOCK_N_dkdv, BLOCK_D],
    )
    V_dq_mask_desc = _maybe_make_tensor_desc(
        V_dq_mask,
        shape=[B, H, S_kv, BLOCK_D],
        strides=[stride_vb, stride_vh, stride_vk, 1],
        block_shape=[1, 1, BLOCK_N_dq_mask, BLOCK_D],
    )
    V_dq_desc = _maybe_make_tensor_desc(
        V_dq,
        shape=[B, H, S_kv, BLOCK_D],
        strides=[stride_vb, stride_vh, stride_vk, 1],
        block_shape=[1, 1, BLOCK_N_dq, BLOCK_D],
    )
    dO_dq_desc = _maybe_make_tensor_desc(
        dO_dq,
        shape=[B, H, S_qo, BLOCK_D],
        strides=[stride_ob, stride_oh, stride_om, 1],
        block_shape=[1, 1, BLOCK_M_dq, BLOCK_D],
    )
    dO_dkdv_mask_desc = _maybe_make_tensor_desc(
        dO_dkdv_mask,
        shape=[B, H, S_qo, BLOCK_D],
        strides=[stride_ob, stride_oh, stride_om, 1],
        block_shape=[1, 1, BLOCK_M_dkdv_mask, BLOCK_D],
    )
    dO_dkdv_desc = _maybe_make_tensor_desc(
        dO_dkdv,
        shape=[B, H, S_qo, BLOCK_D],
        strides=[stride_ob, stride_oh, stride_om, 1],
        block_shape=[1, 1, BLOCK_M_dkdv, BLOCK_D],
    )
    dQ_desc = _maybe_make_tensor_desc(
        dQ,
        shape=[B, H, S_qo, BLOCK_D],
        strides=[stride_qb, stride_qh, stride_qm, 1],
        block_shape=[1, 1, BLOCK_M_dq, BLOCK_D],
    )
    dK_desc = _maybe_make_tensor_desc(
        dK,
        shape=[B, H, S_kv, BLOCK_D],
        strides=[stride_kb, stride_kh, stride_kn, 1],
        block_shape=[1, 1, BLOCK_N_dkdv, BLOCK_D],
    )
    dV_desc = _maybe_make_tensor_desc(
        dV,
        shape=[B, H, S_kv, BLOCK_D],
        strides=[stride_vb, stride_vh, stride_vk, 1],
        block_shape=[1, 1, BLOCK_N_dkdv, BLOCK_D],
    )
    if USE_DESC_FOR_CORR:
        L_dkdv_mask = _maybe_make_tensor_desc(
            L_dkdv_mask,
            shape=[B, H, S_qo],
            strides=[stride_lb, stride_lh, 1],
            block_shape=[1, 1, BLOCK_M_dkdv_mask],
        )
        L_dkdv = _maybe_make_tensor_desc(
            L_dkdv,
            shape=[B, H, S_qo],
            strides=[stride_lb, stride_lh, 1],
            block_shape=[1, 1, BLOCK_M_dkdv],
        )
        L_dq = _maybe_make_tensor_desc(
            L_dq,
            shape=[B, H, S_qo],
            strides=[stride_lb, stride_lh, 1],
            block_shape=[1, 1, BLOCK_N_dq],
        )
        Delta_dkdv_mask = _maybe_make_tensor_desc(
            Delta_dkdv_mask,
            shape=[B, H, S_qo],
            strides=[stride_lb, stride_lh, 1],
            block_shape=[1, 1, BLOCK_M_dkdv_mask],
        )
        Delta_dkdv = _maybe_make_tensor_desc(
            Delta_dkdv,
            shape=[B, H, S_qo],
            strides=[stride_lb, stride_lh, 1],
            block_shape=[1, 1, BLOCK_M_dkdv],
        )
        Delta_dq = _maybe_make_tensor_desc(
            Delta_dq,
            shape=[B, H, S_qo],
            strides=[stride_lb, stride_lh, 1],
            block_shape=[1, 1, BLOCK_N_dq],
        )

    dv = tl.zeros([BLOCK_N_dkdv, BLOCK_D], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N_dkdv, BLOCK_D], dtype=tl.float32)

    k = K_dkdv_desc.load([off_b, off_h, start_n, 0]).reshape(
        BLOCK_N_dkdv, BLOCK_D
    )
    v = V_dkdv_desc.load([off_b, off_h, start_n, 0]).reshape(
        BLOCK_N_dkdv, BLOCK_D
    )

    if IS_CAUSAL:
        num_steps = BLOCK_N_dkdv // BLOCK_M_dkdv_mask
        # Compute dK and dV for masked (diagonal) blocks.
        dk, dv = _attn_bwd_dkdv(
            dk,
            dv,  #
            Q_dkdv_mask_desc,
            k,
            v,
            sm_scale,  #
            dO_dkdv_mask_desc,  #
            L_dkdv_mask,
            Delta_dkdv_mask,  #
            off_b,
            off_h,
            stride_lb,
            stride_lh,
            BLOCK_M_dkdv_mask,
            BLOCK_N_dkdv,
            BLOCK_D,  #
            start_n,
            start_m,
            num_steps,
            dtype,  #
            S_qo,
            USE_DESC_FOR_CORR=USE_DESC_FOR_CORR,
            MASK=True,  #
        )

        start_m += num_steps * BLOCK_M_dkdv_mask
    else:
        start_m = 0
    num_steps = tl.cdiv(S_qo - start_m, BLOCK_M_dkdv)
    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv(  #
        dk,
        dv,  #
        Q_dkdv_desc,
        k,
        v,
        sm_scale,  #
        dO_dkdv_desc,  #
        L_dkdv,
        Delta_dkdv,  #
        off_b,
        off_h,
        stride_lb,
        stride_lh,
        BLOCK_M_dkdv,
        BLOCK_N_dkdv,
        BLOCK_D,  #
        start_n,
        start_m,
        num_steps,
        dtype,  #
        S_qo,
        USE_DESC_FOR_CORR=USE_DESC_FOR_CORR,
        MASK=False,  #
    )

    # Write back dK and dV.
    dK_desc.store(
        [off_b, off_h, start_n, 0],
        dk.reshape(1, 1, BLOCK_N_dkdv, BLOCK_D).to(dtype),
    )
    dV_desc.store(
        [off_b, off_h, start_n, 0],
        dv.reshape(1, 1, BLOCK_N_dkdv, BLOCK_D).to(dtype),
    )

    start_m = pid * BLOCK_M_dq
    end_n = start_m + BLOCK_M_dq

    offset_s_ptrs = start_m + tl.reshape(
        tl.arange(0, BLOCK_M_dq), (1, 1, BLOCK_M_dq)
    )
    offset_ptrs = off_b * stride_lb + off_h * stride_lh + offset_s_ptrs

    q = Q_dq_desc.load([off_b, off_h, start_m, 0]).reshape(BLOCK_M_dq, BLOCK_D)
    dq = tl.zeros([BLOCK_M_dq, BLOCK_D], dtype=tl.float32)
    do = dO_dq_desc.load([off_b, off_h, start_m, 0]).reshape(
        BLOCK_M_dq, BLOCK_D
    )

    if USE_DESC_FOR_CORR:
        m = L_dq.load([off_b, off_h, start_m]).reshape(BLOCK_M_dq)
        di = Delta_dq.load([off_b, off_h, start_m]).reshape(BLOCK_M_dq)
    else:
        m = tl.load(
            L_dq + offset_ptrs, mask=offset_s_ptrs < S_kv, other=0.0
        ).reshape(BLOCK_M_dq)
        di = tl.load(
            Delta_dq + offset_ptrs, mask=offset_s_ptrs < S_kv, other=0.0
        ).reshape(BLOCK_M_dq)

    m = m[:, None]
    di = di[:, None]

    if IS_CAUSAL:
        # Compute dQ for masked (diagonal) blocks.
        # NOTE: This code scans each row of QK^T backward (from right to left,
        # but inside each call to _attn_bwd_dq, from left to right), but that's
        # not due to anything important.  I just wanted to reuse the loop
        # structure for dK & dV above as much as possible.
        num_steps = BLOCK_M_dq // BLOCK_N_dq_mask
        dq = _attn_bwd_dq(
            dq,
            q,
            K_dq_mask_desc,
            V_dq_mask_desc,  #
            do,
            m,
            di,
            sm_scale,  #
            off_b,
            off_h,
            BLOCK_M_dq,
            BLOCK_N_dq_mask,
            BLOCK_D,  #
            start_m,
            end_n - num_steps * BLOCK_N_dq_mask,
            num_steps,
            dtype,  #
            MASK=True,  #
        )
        end_n -= num_steps * BLOCK_N_dq_mask
    else:
        end_n = S_kv
    num_steps = tl.cdiv(end_n, BLOCK_N_dq)
    # Compute dQ for non-masked blocks.
    dq = _attn_bwd_dq(
        dq,
        q,
        K_dq_desc,
        V_dq_desc,  #
        do,
        m,
        di,
        sm_scale,  #
        off_b,
        off_h,
        BLOCK_M_dq,
        BLOCK_N_dq,
        BLOCK_D,  #
        start_m,
        0,
        num_steps,
        dtype,  #
        MASK=False,  #
    )
    # Write back dQ.
    dQ_desc.store(
        [off_b, off_h, start_m, 0],
        dq.reshape(1, 1, BLOCK_M_dq, BLOCK_D).to(dtype),
    )


def _attention_bwd_tma(ctx, do):
    q, k, v, o, l = ctx.saved_tensors
    B, H, S_qo, S_kv = ctx.shapes
    is_causal, BLOCK_D = ctx.launch_configs
    USE_DESC_FOR_CORR = False
    do = do.contiguous()
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    minus_delta = torch.empty_like(l)
    minus_l = torch.empty_like(l)
    assert dq.stride() == q.stride()
    assert dk.stride() == k.stride()
    assert dv.stride() == v.stride()
    assert do.stride() == o.stride()

    if supports_host_descriptor():
        dummy_block_qo = [1, 1, 1, 1]
        dummy_block_kv = [1, 1, 1, 1]
        dummy_block_ld = [1, 1, 1]
        dO_desc = TensorDescriptor(
            do,
            shape=[B, H, S_qo, BLOCK_D],
            strides=[o.stride(0), o.stride(1), o.stride(2), 1],
            block_shape=dummy_block_qo,
        )
        O_desc = TensorDescriptor(
            o,
            shape=[B, H, S_qo, BLOCK_D],
            strides=[o.stride(0), o.stride(1), o.stride(2), 1],
            block_shape=dummy_block_qo,
        )
        dQ_desc = TensorDescriptor(
            dq,
            shape=[B, H, S_qo, BLOCK_D],
            strides=[dq.stride(0), dq.stride(1), dq.stride(2), 1],
            block_shape=dummy_block_qo,
        )
        dK_desc = TensorDescriptor(
            dk,
            shape=[B, H, S_kv, BLOCK_D],
            strides=[dk.stride(0), dk.stride(1), dk.stride(2), 1],
            block_shape=dummy_block_kv,
        )
        dV_desc = TensorDescriptor(
            dv,
            shape=[B, H, S_kv, BLOCK_D],
            strides=[dv.stride(0), dv.stride(1), dv.stride(2), 1],
            block_shape=dummy_block_kv,
        )
        if supports_tma(l):
            L_desc = TensorDescriptor(
                l,
                shape=[B, H, S_qo],
                strides=[l.stride(0), l.stride(1), 1],
                block_shape=dummy_block_ld,
            )
            minus_L_desc = TensorDescriptor(
                minus_l,
                shape=[B, H, S_qo],
                strides=[minus_l.stride(0), minus_l.stride(1), 1],
                block_shape=dummy_block_ld,
            )
            Delta_desc = TensorDescriptor(
                minus_delta,
                shape=[B, H, S_qo],
                strides=[minus_delta.stride(0), minus_delta.stride(1), 1],
                block_shape=dummy_block_ld,
            )
            USE_DESC_FOR_CORR = True
        else:
            L_desc = l
            minus_L_desc = minus_l
            Delta_desc = minus_delta
    else:
        dO_desc = do
        O_desc = o
        dQ_desc = dq
        dK_desc = dk
        dV_desc = dv
        L_desc = l
        minus_L_desc = minus_l
        Delta_desc = minus_delta

    grid = lambda args: (triton.cdiv(S_qo, args['BLOCK_M']), B * H, 1)
    fmha_bwd_preprocess_kernel[grid](
        O_desc,
        dO_desc,
        L_desc,
        Delta_desc,
        minus_L_desc,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        l.stride(0),
        l.stride(1),
        B,
        H,
        S_qo,
        ctx.sm_scale,
        BLOCK_D=BLOCK_D,
        USE_DESC_FOR_CORR=USE_DESC_FOR_CORR,
    )
    if supports_host_descriptor():
        Q_desc_dq = TensorDescriptor(
            q,
            shape=[B, H, S_qo, BLOCK_D],
            strides=[q.stride(0), q.stride(1), q.stride(2), 1],
            block_shape=dummy_block_qo,
        )
        Q_desc_dkdv_mask = TensorDescriptor(
            q,
            shape=[B, H, S_qo, BLOCK_D],
            strides=[q.stride(0), q.stride(1), q.stride(2), 1],
            block_shape=dummy_block_qo,
        )
        Q_desc_dkdv = TensorDescriptor(
            q,
            shape=[B, H, S_qo, BLOCK_D],
            strides=[q.stride(0), q.stride(1), q.stride(2), 1],
            block_shape=dummy_block_qo,
        )
        K_desc_dkdv = TensorDescriptor(
            k,
            shape=[B, H, S_kv, BLOCK_D],
            strides=[k.stride(0), k.stride(1), k.stride(2), 1],
            block_shape=dummy_block_kv,
        )
        K_desc_dq_mask = TensorDescriptor(
            k,
            shape=[B, H, S_kv, BLOCK_D],
            strides=[k.stride(0), k.stride(1), k.stride(2), 1],
            block_shape=dummy_block_kv,
        )
        K_desc_dq = TensorDescriptor(
            k,
            shape=[B, H, S_kv, BLOCK_D],
            strides=[k.stride(0), k.stride(1), k.stride(2), 1],
            block_shape=dummy_block_kv,
        )
        V_desc_dkdv = TensorDescriptor(
            v,
            shape=[B, H, S_kv, BLOCK_D],
            strides=[v.stride(0), v.stride(1), v.stride(2), 1],
            block_shape=dummy_block_kv,
        )
        V_desc_dq_mask = TensorDescriptor(
            v,
            shape=[B, H, S_kv, BLOCK_D],
            strides=[v.stride(0), v.stride(1), v.stride(2), 1],
            block_shape=dummy_block_kv,
        )
        V_desc_dq = TensorDescriptor(
            v,
            shape=[B, H, S_kv, BLOCK_D],
            strides=[v.stride(0), v.stride(1), v.stride(2), 1],
            block_shape=dummy_block_kv,
        )
        dO_desc_dq = TensorDescriptor(
            do,
            shape=[B, H, S_qo, BLOCK_D],
            strides=[do.stride(0), do.stride(1), do.stride(2), 1],
            block_shape=dummy_block_qo,
        )
        dO_desc_dkdv_mask = TensorDescriptor(
            do,
            shape=[B, H, S_qo, BLOCK_D],
            strides=[do.stride(0), do.stride(1), do.stride(2), 1],
            block_shape=dummy_block_qo,
        )
        dO_desc_dkdv = TensorDescriptor(
            do,
            shape=[B, H, S_qo, BLOCK_D],
            strides=[do.stride(0), do.stride(1), do.stride(2), 1],
            block_shape=dummy_block_qo,
        )
    else:
        Q_desc_dq = Q_desc_dkdv_mask = Q_desc_dkdv = q
        K_desc_dkdv = K_desc_dq_mask = K_desc_dq = k
        V_desc_dkdv = V_desc_dq_mask = V_desc_dq = v
        dO_desc_dq = dO_desc_dkdv_mask = dO_desc_dkdv = do

    USE_DESC_FOR_CORR = False
    if supports_host_descriptor() and supports_tma(minus_l):
        minus_L_desc_dq = TensorDescriptor(
            minus_l,
            shape=[B, H, S_qo],
            strides=[minus_l.stride(0), minus_l.stride(1), 1],
            block_shape=dummy_block_ld,
        )
        minus_L_desc_dkdv_mask = TensorDescriptor(
            minus_l,
            shape=[B, H, S_qo],
            strides=[minus_l.stride(0), minus_l.stride(1), 1],
            block_shape=dummy_block_ld,
        )
        minus_L_desc_dkdv = TensorDescriptor(
            minus_l,
            shape=[B, H, S_qo],
            strides=[minus_l.stride(0), minus_l.stride(1), 1],
            block_shape=dummy_block_ld,
        )
        Delta_desc_dq = TensorDescriptor(
            minus_delta,
            shape=[B, H, S_qo],
            strides=[minus_delta.stride(0), minus_delta.stride(1), 1],
            block_shape=dummy_block_ld,
        )
        Delta_desc_dkdv_mask = TensorDescriptor(
            minus_delta,
            shape=[B, H, S_qo],
            strides=[minus_delta.stride(0), minus_delta.stride(1), 1],
            block_shape=dummy_block_ld,
        )
        Delta_desc_dkdv = TensorDescriptor(
            minus_delta,
            shape=[B, H, S_qo],
            strides=[minus_delta.stride(0), minus_delta.stride(1), 1],
            block_shape=dummy_block_ld,
        )
        USE_DESC_FOR_CORR = True
    else:
        minus_L_desc_dq = (
            minus_L_desc_dkdv_mask
        ) = minus_L_desc_dkdv = minus_l
        Delta_desc_dq = (
            Delta_desc_dkdv_mask
        ) = Delta_desc_dkdv = minus_delta

    grid = lambda args: (triton.cdiv(S_kv, args['BLOCK_N_dkdv']), B * H, 1)
    fmha_bwd_two_loop_kernel[grid](
        Q_desc_dq,
        Q_desc_dkdv_mask,
        Q_desc_dkdv,
        K_desc_dkdv,
        K_desc_dq_mask,
        K_desc_dq,
        V_desc_dkdv,
        V_desc_dq_mask,
        V_desc_dq,
        dO_desc_dq,
        dO_desc_dkdv_mask,
        dO_desc_dkdv,
        dQ_desc,
        dK_desc,
        dV_desc,
        minus_L_desc_dq,
        minus_L_desc_dkdv_mask,
        minus_L_desc_dkdv,
        Delta_desc_dq,
        Delta_desc_dkdv_mask,
        Delta_desc_dkdv,
        ctx.sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        l.stride(0),
        l.stride(1),
        B,
        H,
        S_qo,
        S_kv,
        BLOCK_D=BLOCK_D,
        IS_CAUSAL=is_causal,
        USE_DESC_FOR_CORR=USE_DESC_FOR_CORR,
        dtype=q.dtype,
    )

    return dq, dk, dv, None, None, None, None, None, None, None
