"""Flash Attention 2 implementation using Triton kernels."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES: tl.constexpr, N_KEYS: tl.constexpr, D: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    scale: tl.constexpr,
):
    batch_idx = tl.program_id(1)
    q_tile_idx = tl.program_id(0)

    # Offsets for this Q tile
    q_start = q_tile_idx * BLOCK_Q
    q_offsets = q_start + tl.arange(0, BLOCK_Q)
    d_offsets = tl.arange(0, D)

    # Initialize accumulators
    m_i = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    o_i = tl.zeros((BLOCK_Q, D), dtype=tl.float32)

    # Load Q tile: (BLOCK_Q, D)
    q_ptrs = Q_ptr + batch_idx * stride_qb + q_offsets[:, None] * stride_qq + d_offsets[None, :] * stride_qd
    q_mask = (q_offsets[:, None] < N_QUERIES) & (d_offsets[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    # Determine KV range
    if IS_CAUSAL:
        kv_end = tl.minimum(N_KEYS, q_start + BLOCK_Q)
    else:
        kv_end = N_KEYS

    num_kv_tiles = tl.cdiv(kv_end, BLOCK_KV)

    for j in range(num_kv_tiles):
        kv_start = j * BLOCK_KV
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)

        # Load K tile: (BLOCK_KV, D)
        k_ptrs = K_ptr + batch_idx * stride_kb + kv_offsets[:, None] * stride_kk + d_offsets[None, :] * stride_kd
        k_mask = (kv_offsets[:, None] < N_KEYS) & (d_offsets[None, :] < D)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # Load V tile: (BLOCK_KV, D)
        v_ptrs = V_ptr + batch_idx * stride_vb + kv_offsets[:, None] * stride_vk + d_offsets[None, :] * stride_vd
        v_mask = (kv_offsets[:, None] < N_KEYS) & (d_offsets[None, :] < D)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

        # S = Q @ K^T * scale: (BLOCK_Q, BLOCK_KV)
        s = tl.dot(q, tl.trans(k)) * scale

        # Apply causal mask
        if IS_CAUSAL:
            causal_mask = q_offsets[:, None] >= kv_offsets[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # Apply boundary mask for out-of-bounds keys
        boundary_mask = kv_offsets[None, :] < N_KEYS
        s = tl.where(boundary_mask, s, float("-inf"))

        # Online softmax
        m_ij = tl.max(s, axis=1)  # (BLOCK_Q,)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        p = tl.exp(s - m_new[:, None])

        l_i = alpha * l_i + tl.sum(p, axis=1)
        o_i = alpha[:, None] * o_i + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # Normalize
    o_i = o_i / l_i[:, None]
    lse = m_i + tl.log(l_i)

    # Store output
    o_ptrs = O_ptr + batch_idx * stride_ob + q_offsets[:, None] * stride_oq + d_offsets[None, :] * stride_od
    o_mask = (q_offsets[:, None] < N_QUERIES) & (d_offsets[None, :] < D)
    tl.store(o_ptrs, o_i.to(tl.float32), mask=o_mask)

    # Store LSE
    l_ptrs = L_ptr + batch_idx * stride_lb + q_offsets * stride_lq
    l_mask = q_offsets < N_QUERIES
    tl.store(l_ptrs, lse, mask=l_mask)


class FlashAttentionTriton(torch.autograd.Function):
    """FlashAttention2 using Triton kernels for forward, torch.compile for backward."""

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        batch_size, n_queries, d = q.shape
        n_keys = k.shape[1]
        scale = 1.0 / math.sqrt(d)

        # Ensure contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        o = torch.empty_like(q)
        l = torch.empty(batch_size, n_queries, device=q.device, dtype=q.dtype)

        BLOCK_Q = min(64, triton.next_power_of_2(n_queries))
        BLOCK_KV = min(64, triton.next_power_of_2(n_keys))
        # D must be power of 2 for tl.dot
        D_PADDED = triton.next_power_of_2(d)

        grid = (triton.cdiv(n_queries, BLOCK_Q), batch_size)

        _flash_attn_fwd_kernel[grid](
            q, k, v, o, l,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            l.stride(0), l.stride(1),
            N_QUERIES=n_queries, N_KEYS=n_keys, D=D_PADDED,
            BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV,
            IS_CAUSAL=is_causal,
            scale=scale,
        )

        ctx.save_for_backward(q, k, v, o, l)
        ctx.is_causal = is_causal

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l = ctx.saved_tensors
        is_causal = ctx.is_causal
        # Use the PyTorch backward implementation for correctness
        dq, dk, dv = _flash_attn_backward_pytorch(q, k, v, o, l, do, is_causal)
        return dq, dk, dv, None


def _flash_attn_backward_pytorch(q, k, v, o, l, do, is_causal):
    """Backward pass implemented in PyTorch (usable with torch.compile)."""
    batch_size, n_queries, d = q.shape
    n_keys = k.shape[1]
    scale = 1.0 / math.sqrt(d)

    Br = min(64, n_queries)
    Bc = min(64, n_keys)
    Tr = math.ceil(n_queries / Br)
    Tc = math.ceil(n_keys / Bc)

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    D = (o * do).sum(dim=-1)  # (batch, n_queries)

    for j in range(Tc):
        kj = k[:, j * Bc : (j + 1) * Bc, :]
        vj = v[:, j * Bc : (j + 1) * Bc, :]
        dkj = torch.zeros_like(kj)
        dvj = torch.zeros_like(vj)

        min_i = 0
        if is_causal:
            first_key_pos = j * Bc
            min_i = first_key_pos // Br

        for i in range(min_i, Tr):
            qi = q[:, i * Br : (i + 1) * Br, :]
            doi = do[:, i * Br : (i + 1) * Br, :]
            li = l[:, i * Br : (i + 1) * Br]
            di = D[:, i * Br : (i + 1) * Br]

            s = torch.bmm(qi, kj.transpose(-2, -1)) * scale

            if is_causal:
                q_indices = torch.arange(i * Br, i * Br + qi.shape[1], device=q.device)
                k_indices = torch.arange(j * Bc, j * Bc + kj.shape[1], device=q.device)
                causal_mask = q_indices[:, None] >= k_indices[None, :]
                s = s.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

            p = torch.exp(s - li.unsqueeze(-1))
            dvj = dvj + torch.bmm(p.transpose(-2, -1), doi)
            dp = torch.bmm(doi, vj.transpose(-2, -1))
            ds = p * (dp - di.unsqueeze(-1))
            dq[:, i * Br : (i + 1) * Br, :] += torch.bmm(ds, kj) * scale
            dkj = dkj + torch.bmm(ds.transpose(-2, -1), qi) * scale

        dk[:, j * Bc : (j + 1) * Bc, :] = dkj
        dv[:, j * Bc : (j + 1) * Bc, :] = dvj

    return dq, dk, dv
