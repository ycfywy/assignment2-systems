"""Optimizer State Sharding implementation."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """
    An optimizer wrapper that shards optimizer state across ranks.

    Each rank only maintains optimizer state for a subset of parameters
    (approximately 1/world_size). After each optimizer step, updated
    parameters are broadcast to all other ranks.
    """

    def __init__(self, params, optimizer_cls: type[Optimizer], **kwargs: Any):
        # Materialize params into a list (handle generators)
        params = list(params)

        # De-duplicate parameters (handle tied weights)
        seen = set()
        unique_params = []
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                unique_params.append(p)

        self._all_params = unique_params
        self._optimizer_cls = optimizer_cls
        self._optimizer_kwargs = kwargs

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Shard parameters across ranks
        # Each rank is responsible for ~1/world_size of all parameters
        self._rank_to_params: dict[int, list[torch.nn.Parameter]] = {r: [] for r in range(world_size)}
        for i, p in enumerate(unique_params):
            owner_rank = i % world_size
            self._rank_to_params[owner_rank].append(p)

        self._my_params = self._rank_to_params[rank]

        # Create the actual optimizer only for this rank's parameters
        if len(self._my_params) > 0:
            self._inner_optimizer = optimizer_cls(self._my_params, **kwargs)
        else:
            self._inner_optimizer = None

        # Initialize the super class with all parameters but empty defaults
        # We need to pass the params through the super constructor
        defaults = kwargs.copy()
        super().__init__(unique_params, defaults)

    def step(self, closure=None, **kwargs):
        """
        Perform an optimization step on the local shard, then broadcast
        updated parameters to all other ranks.
        """
        loss = None
        if self._inner_optimizer is not None:
            loss = self._inner_optimizer.step(closure=closure, **kwargs)

        # Broadcast updated parameters from each owning rank
        world_size = dist.get_world_size()
        for owner_rank in range(world_size):
            for p in self._rank_to_params[owner_rank]:
                dist.broadcast(p.data, src=owner_rank)

        return loss

    def zero_grad(self, set_to_none: bool = True):
        """Zero out gradients for all parameters."""
        for p in self._all_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    @torch.no_grad()
    def add_param_group(self, param_group: dict[str, Any]):
        """Add a parameter group. Called by super().__init__."""
        # The super().__init__ calls add_param_group for each param group.
        # We just let it pass through to the base Optimizer.
        super().add_param_group(param_group)
