"""Benchmarking script for profiling Transformer model performance."""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn


MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B":    {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}


def create_model(size: str, context_length: int = 512, vocab_size: int = 50257, device: str = "cuda", compile_model: bool = False):
    """Create a BasicsTransformerLM model of the given size."""
    from cs336_basics.model import BasicsTransformerLM

    config = MODEL_CONFIGS[size]
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        **config,
    ).to(device)

    if compile_model:
        model = torch.compile(model)

    return model


def benchmark(
    model: nn.Module,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: str,
    warmup_steps: int = 3,
    num_steps: int = 10,
    mode: str = "full",  # "forward", "forward_backward", "full"
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
):
    """
    Benchmark the model.

    Args:
        mode: "forward" - only forward pass
              "forward_backward" - forward + backward
              "full" - forward + backward + optimizer step
    """
    optimizer = None
    if mode == "full":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    loss_fn = nn.CrossEntropyLoss()

    # Warmup
    for _ in range(warmup_steps):
        x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

        if optimizer:
            optimizer.zero_grad()

        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))

        if mode in ("forward_backward", "full"):
            loss.backward()

        if optimizer:
            optimizer.step()

        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_steps):
        x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

        if optimizer:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))

        if mode in ("forward_backward", "full"):
            loss.backward()

        if optimizer:
            optimizer.step()

        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    return avg_time, times


def get_peak_memory(device: str = "cuda"):
    """Get peak GPU memory allocated in MiB."""
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Transformer models")
    parser.add_argument("--size", type=str, default="small", choices=MODEL_CONFIGS.keys())
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--mode", type=str, default="full", choices=["forward", "forward_backward", "full"])
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--use-amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16

    print(f"Creating {args.size} model...")
    model = create_model(
        args.size, args.context_length, args.vocab_size,
        args.device, args.compile
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params * 4 / 1024**3:.2f} GiB in fp32)")

    torch.cuda.reset_peak_memory_stats()

    avg_time, times = benchmark(
        model, args.batch_size, args.context_length, args.vocab_size,
        args.device, args.warmup_steps, args.num_steps,
        args.mode, args.use_amp, amp_dtype,
    )

    peak_mem = get_peak_memory(args.device)

    print(f"\n{'='*50}")
    print(f"Model size: {args.size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Context length: {args.context_length}")
    print(f"Mode: {args.mode}")
    print(f"Compile: {args.compile}")
    print(f"AMP: {args.use_amp} ({args.amp_dtype})")
    print(f"Average time: {avg_time*1000:.2f} ms")
    print(f"Peak GPU memory: {peak_mem:.2f} MiB ({peak_mem/1024:.2f} GiB)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
