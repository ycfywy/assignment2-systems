from __future__ import annotations

import torch


def get_flashattention_autograd_function_pytorch() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using only standard PyTorch operations (no Triton!).
    """
    from cs336_systems.flash_attention import FlashAttentionPyTorch
    return FlashAttentionPyTorch


def get_flashattention_autograd_function_triton() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    """
    from cs336_systems.flash_attention_triton import FlashAttentionTriton
    return FlashAttentionTriton


def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a DDP container that handles parameter broadcasting and
    gradient synchronization with overlapping communication.
    """
    from cs336_systems.ddp import DDP
    return DDP(module)


def ddp_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after backward pass, before optimizer step.
    """
    ddp_model.finish_gradient_synchronization()


def get_fsdp(module: torch.nn.Module, compute_dtype: torch.dtype | None = None) -> torch.nn.Module:
    """
    Returns an FSDP container for fully-sharded data parallel training.
    """
    from cs336_systems.fsdp import FSDP
    return FSDP(module, compute_dtype=compute_dtype)


def fsdp_on_after_backward(fsdp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after backward pass, before optimizer step for FSDP.
    """
    fsdp_model.finish_gradient_synchronization()


def fsdp_gather_full_params(fsdp_model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    All-gather sharded parameters from the FSDP model to reconstruct full
    parameter tensors.
    """
    return fsdp_model.gather_full_params()


def get_sharded_optimizer(params, optimizer_cls: type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    Returns an optimizer that handles optimizer state sharding.
    """
    from cs336_systems.sharded_optimizer import ShardedOptimizer
    return ShardedOptimizer(params, optimizer_cls, **kwargs)
