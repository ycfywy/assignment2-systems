"""Flash Attention 2 implementation in pure PyTorch."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class FlashAttentionPyTorch(torch.autograd.Function):
    """FlashAttention2 implemented with standard PyTorch operations."""

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        """
        Args:
            q: (batch, n_queries, d)
            k: (batch, n_keys, d)
            v: (batch, n_keys, d)
            is_causal: bool
        Returns:
            o: (batch, n_queries, d)
        """
        batch_size, n_queries, d = q.shape
        n_keys = k.shape[1]
        scale = 1.0 / math.sqrt(d)

        # Tile sizes
        Br = min(64, n_queries)
        Bc = min(64, n_keys)

        # Initialize output, log-sum-exp, and max
        o = torch.zeros_like(q)
        l = torch.zeros(batch_size, n_queries, device=q.device, dtype=q.dtype)
        m = torch.full((batch_size, n_queries), float("-inf"), device=q.device, dtype=q.dtype)

        # Number of tiles
        Tr = math.ceil(n_queries / Br)
        Tc = math.ceil(n_keys / Bc)

        for i in range(Tr):
            qi = q[:, i * Br : (i + 1) * Br, :]  # (batch, Br, d)
            oi = torch.zeros_like(qi)
            li = torch.zeros(batch_size, qi.shape[1], device=q.device, dtype=q.dtype)
            mi = torch.full((batch_size, qi.shape[1]), float("-inf"), device=q.device, dtype=q.dtype)

            # Determine the range of KV tiles to iterate (for causal, skip tiles that are entirely masked)
            max_j = Tc
            if is_causal:
                # The last query in this tile is at position min((i+1)*Br, n_queries) - 1
                last_query_pos = min((i + 1) * Br, n_queries) - 1
                # We only need KV tiles where the first key position <= last_query_pos
                max_j = min(Tc, (last_query_pos // Bc) + 1)

            for j in range(max_j):
                kj = k[:, j * Bc : (j + 1) * Bc, :]  # (batch, Bc, d)
                vj = v[:, j * Bc : (j + 1) * Bc, :]  # (batch, Bc, d)

                # Compute attention scores: (batch, Br, Bc)
                s = torch.bmm(qi, kj.transpose(-2, -1)) * scale

                # Apply causal mask if needed
                if is_causal:
                    q_indices = torch.arange(i * Br, i * Br + qi.shape[1], device=q.device)
                    k_indices = torch.arange(j * Bc, j * Bc + kj.shape[1], device=q.device)
                    causal_mask = q_indices[:, None] >= k_indices[None, :]  # (Br, Bc)
                    s = s.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

                # Online softmax update
                mij = s.max(dim=-1).values  # (batch, Br)
                mi_new = torch.maximum(mi, mij)

                # Correction factors
                alpha = torch.exp(mi - mi_new)  # (batch, Br)
                beta = torch.exp(mij - mi_new)  # (batch, Br)

                # Compute exp(s - mi_new)
                p = torch.exp(s - mi_new.unsqueeze(-1))  # (batch, Br, Bc)

                # Update running sum
                li = alpha * li + p.sum(dim=-1)  # (batch, Br)

                # Update output
                oi = alpha.unsqueeze(-1) * oi + torch.bmm(p, vj)  # (batch, Br, d)

                mi = mi_new

            # Normalize output
            oi = oi / li.unsqueeze(-1)

            # Store results
            o[:, i * Br : (i + 1) * Br, :] = oi
            l[:, i * Br : (i + 1) * Br] = mi + torch.log(li)
            m[:, i * Br : (i + 1) * Br] = mi

        ctx.save_for_backward(q, k, v, o, l)
        ctx.is_causal = is_causal

        return o

    @staticmethod
    def backward(ctx, do):
        """
        Args:
            do: (batch, n_queries, d) - gradient of output
        Returns:
            dq, dk, dv, None (for is_causal)
        """
        q, k, v, o, l = ctx.saved_tensors
        is_causal = ctx.is_causal

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

        # D_i = rowsum(o_i * do_i)
        D = (o * do).sum(dim=-1)  # (batch, n_queries)

        for j in range(Tc):
            kj = k[:, j * Bc : (j + 1) * Bc, :]
            vj = v[:, j * Bc : (j + 1) * Bc, :]
            dkj = torch.zeros_like(kj)
            dvj = torch.zeros_like(vj)

            # Determine Q tile range (for causal, skip tiles where all queries < first key)
            min_i = 0
            if is_causal:
                first_key_pos = j * Bc
                min_i = first_key_pos // Br

            for i in range(min_i, Tr):
                qi = q[:, i * Br : (i + 1) * Br, :]
                oi = o[:, i * Br : (i + 1) * Br, :]
                doi = do[:, i * Br : (i + 1) * Br, :]
                li = l[:, i * Br : (i + 1) * Br]
                di = D[:, i * Br : (i + 1) * Br]

                # Recompute attention scores
                s = torch.bmm(qi, kj.transpose(-2, -1)) * scale  # (batch, Br, Bc)

                if is_causal:
                    q_indices = torch.arange(i * Br, i * Br + qi.shape[1], device=q.device)
                    k_indices = torch.arange(j * Bc, j * Bc + kj.shape[1], device=q.device)
                    causal_mask = q_indices[:, None] >= k_indices[None, :]
                    s = s.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

                # p = exp(s - l) which gives the attention weights
                p = torch.exp(s - li.unsqueeze(-1))  # (batch, Br, Bc)

                # dV += P^T @ dO
                dvj = dvj + torch.bmm(p.transpose(-2, -1), doi)

                # dP = dO @ V^T
                dp = torch.bmm(doi, vj.transpose(-2, -1))  # (batch, Br, Bc)

                # dS = P * (dP - D_i)
                ds = p * (dp - di.unsqueeze(-1))  # (batch, Br, Bc)

                # dQ += dS @ K * scale
                dq[:, i * Br : (i + 1) * Br, :] += torch.bmm(ds, kj) * scale

                # dK += dS^T @ Q * scale
                dkj = dkj + torch.bmm(ds.transpose(-2, -1), qi) * scale

            dk[:, j * Bc : (j + 1) * Bc, :] = dkj
            dv[:, j * Bc : (j + 1) * Bc, :] = dvj

        return dq, dk, dv, None
