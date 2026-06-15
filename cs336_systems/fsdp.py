"""Fully Sharded Data Parallel (FSDP) implementation."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class FSDP(nn.Module):
    """
    Fully Sharded Data Parallel wrapper.

    Shards Linear/Embedding weight parameters across ranks.
    Before forward/backward, all-gathers the full parameters.
    After backward, reduce-scatters gradients so each rank only
    gets the gradient shard corresponding to its parameter shard.

    Non-Linear/Embedding parameters (e.g. RMSNorm) are replicated
    and their gradients are all-reduced.
    """

    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        from cs336_basics.model import Embedding, Linear

        # Categorize parameters
        self._sharded_param_info: list[dict] = []
        self._replicated_params: list[tuple[str, nn.Parameter]] = []
        self._grad_handles: list = []
        self._is_unsharded = False

        seen_param_ids = set()

        for mod_name, mod in module.named_modules():
            if isinstance(mod, (Linear, Embedding)):
                param = mod.weight
                param_name = mod_name + ".weight" if mod_name else "weight"
                if id(param) in seen_param_ids:
                    continue
                seen_param_ids.add(id(param))

                full_shape = param.data.shape
                numel = param.data.numel()
                padded = numel + (self.world_size - numel % self.world_size) % self.world_size
                shard_size = padded // self.world_size

                # Flatten, pad, shard — master weights in fp32
                flat = param.data.float().flatten()
                if flat.numel() < padded:
                    flat = torch.nn.functional.pad(flat, (0, padded - flat.numel()))
                shard = flat[self.rank * shard_size : (self.rank + 1) * shard_size].clone()

                # Replace param data with shard
                param.data = shard
                # Allow any grad dtype (PyTorch 2.11+)
                if hasattr(param, "grad_dtype"):
                    param.grad_dtype = None

                info = {
                    "param_name": param_name,
                    "param": param,
                    "mod": mod,
                    "full_shape": full_shape,
                    "numel": numel,
                    "padded": padded,
                    "shard_size": shard_size,
                }
                self._sharded_param_info.append(info)
            else:
                for pname, param in mod.named_parameters(recurse=False):
                    full_name = mod_name + "." + pname if mod_name else pname
                    if id(param) in seen_param_ids:
                        continue
                    seen_param_ids.add(id(param))
                    self._replicated_params.append((full_name, param))

        # Build lookup from param id -> info for sharded params
        self._param_id_to_info = {id(info["param"]): info for info in self._sharded_param_info}

        # Register gradient hooks
        for info in self._sharded_param_info:
            info["param"].register_post_accumulate_grad_hook(
                self._make_sharded_grad_hook(info)
            )

        for _, param in self._replicated_params:
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self._make_replicated_grad_hook()
                )

    def _make_sharded_grad_hook(self, info):
        """Reduce gradient for a sharded parameter (gloo-compatible)."""
        def hook(p):
            if p.grad is None:
                return
            # Gradient is w.r.t. the full (unsharded) parameter
            grad_flat = p.grad.float().flatten()
            padded = info["padded"]
            shard_size = info["shard_size"]

            if grad_flat.numel() < padded:
                grad_flat = torch.nn.functional.pad(grad_flat, (0, padded - grad_flat.numel()))

            # Use all_reduce instead of reduce_scatter for gloo compatibility
            handle = dist.all_reduce(grad_flat, op=dist.ReduceOp.SUM, async_op=True)
            self._grad_handles.append((handle, p, grad_flat, "sharded", info["param_name"]))
        return hook

    def _make_replicated_grad_hook(self):
        """All-reduce gradient for a replicated parameter."""
        def hook(p):
            if p.grad is None:
                return
            p.grad = p.grad.float()
            handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._grad_handles.append((handle, p, None, "replicated", None))
        return hook

    def _all_gather_params(self):
        """All-gather sharded params to get full params for compute."""
        if self._is_unsharded:
            return
        self._saved_shards: dict[str, torch.Tensor] = {}

        for info in self._sharded_param_info:
            param = info["param"]
            full_shape = info["full_shape"]
            shard_size = info["shard_size"]

            self._saved_shards[info["param_name"]] = param.data.clone()

            gathered = [torch.zeros(shard_size, device=param.device, dtype=torch.float32)
                        for _ in range(self.world_size)]
            dist.all_gather(gathered, param.data.float())
            full_flat = torch.cat(gathered)[:info["numel"]]
            full_param = full_flat.reshape(full_shape)

            if self.compute_dtype is not None:
                full_param = full_param.to(self.compute_dtype)

            param.data = full_param

        self._is_unsharded = True

    def _reshard_params(self):
        """Restore sharded parameters."""
        if not self._is_unsharded:
            return
        for info in self._sharded_param_info:
            param_name = info["param_name"]
            if param_name in self._saved_shards:
                info["param"].data = self._saved_shards[param_name]
        self._saved_shards.clear()
        self._is_unsharded = False

    def forward(self, *inputs, **kwargs):
        self._all_gather_params()
        output = self.module(*inputs, **kwargs)

        if torch.is_grad_enabled():
            output.register_hook(self._on_backward_start)

        self._reshard_params()
        return output

    def _on_backward_start(self, grad_output):
        """Called when backward pass begins."""
        self._all_gather_params()
        return grad_output

    def finish_gradient_synchronization(self):
        """Wait for async gradient ops, reshard params, and set gradient shards."""
        # First, reshard params back from full to shard
        # (backward left them unsharded)
        self._reshard_params()

        # Now wait for gradient communication and assign grad shards
        for handle, param, grad_data, kind, param_name in self._grad_handles:
            handle.wait()
            if kind == "sharded":
                # grad_data is the full all-reduced flat gradient; take this rank's shard
                info = self._param_id_to_info[id(param)]
                shard_size = info["shard_size"]
                grad_shard = grad_data[self.rank * shard_size : (self.rank + 1) * shard_size]
                param.grad = (grad_shard / self.world_size).to(param.data.dtype)
            else:
                if param.grad is not None:
                    param.grad = (param.grad / self.world_size).to(param.data.dtype)
        self._grad_handles.clear()

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """All-gather to reconstruct full parameter state dict."""
        result = {}

        for info in self._sharded_param_info:
            param = info["param"]
            full_shape = info["full_shape"]
            shard_size = info["shard_size"]

            gathered = [torch.zeros(shard_size, device=param.device, dtype=torch.float32)
                        for _ in range(self.world_size)]
            dist.all_gather(gathered, param.data.float())
            full_flat = torch.cat(gathered)[:info["numel"]]
            result[info["param_name"]] = full_flat.reshape(full_shape)

        for param_name, param in self._replicated_params:
            result[param_name] = param.data.clone()

        return result

    def named_parameters(self, *args, **kwargs):
        return self.module.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.module.parameters(*args, **kwargs)
