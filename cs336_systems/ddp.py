"""Distributed Data Parallel (DDP) implementation with overlapping communication."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class DDP(nn.Module):
    """
    Distributed Data Parallel wrapper that overlaps backward computation
    with gradient communication by asynchronously all-reducing each parameter's
    gradient as soon as it's ready.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._handles: list = []

        # Broadcast parameters from rank 0 to all other ranks
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Track which parameters we've already registered hooks for
        # (handles tied weights - same param object used in multiple places)
        seen_params = set()

        # Register gradient hooks for async all-reduce
        for param in self.module.parameters():
            if not param.requires_grad:
                continue
            if id(param) in seen_params:
                continue
            seen_params.add(id(param))
            param.register_post_accumulate_grad_hook(self._make_grad_hook())

    def _make_grad_hook(self):
        def hook(param):
            if param.grad is not None:
                # Average gradients across all ranks
                handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
                self._handles.append((handle, param))
        return hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """Wait for all async all-reduce operations to complete and average gradients."""
        world_size = dist.get_world_size()
        for handle, param in self._handles:
            handle.wait()
            if param.grad is not None:
                param.grad.div_(world_size)
        self._handles.clear()

    def named_parameters(self, *args, **kwargs):
        return self.module.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.module.parameters(*args, **kwargs)
