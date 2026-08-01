"""FlashAttention varlen compatibility implemented with fused PyTorch SDPA."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


_FUSED_BACKENDS = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
]


def _fused_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float,
    softmax_scale: float | None,
    causal: bool,
) -> torch.Tensor:
    """Run SDPA without permitting the O(N^2)-memory math implementation."""

    if q.device.type != "cuda":
        raise RuntimeError("PTv3 flash compatibility requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"Fused PTv3 attention requires float16/bfloat16, got {q.dtype}. "
            "Run Pointcept with AMP enabled."
        )

    # Listing only fused backends means an unsupported shape fails loudly
    # instead of silently falling back to the memory-hungry math kernel.
    with sdpa_kernel(_FUSED_BACKENDS):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )


def _uniform_varlen_attention(
    qkv: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    *,
    dropout_p: float,
    softmax_scale: float | None,
    causal: bool,
) -> torch.Tensor:
    _, _, heads, head_dim = qkv.shape
    packed = qkv.reshape(batch_size, sequence_length, 3, heads, head_dim)
    q, k, v = packed.unbind(dim=2)
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    out = _fused_sdpa(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    return out.transpose(1, 2).reshape(-1, heads, head_dim)


def _variable_attention(
    qkv: torch.Tensor,
    boundaries: Sequence[int],
    *,
    dropout_p: float,
    softmax_scale: float | None,
    causal: bool,
) -> torch.Tensor:
    # PTv3 normally pads all serialized patches to an equal size, so this is a
    # correctness fallback for unusual callers. Every segment still uses a
    # fused kernel and remains memory efficient, albeit with more launches.
    outputs: list[torch.Tensor] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        length = end - start
        if length <= 0:
            continue
        outputs.append(
            _uniform_varlen_attention(
                qkv[start:end],
                1,
                length,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
            )
        )
    if not outputs:
        return qkv.new_empty((0, qkv.shape[2], qkv.shape[3]))
    return torch.cat(outputs, dim=0)


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    *args,
    **kwargs,
) -> torch.Tensor:
    """PTv3-compatible subset of FlashAttention's packed varlen operation.

    Args follow FlashAttention-2. ``qkv`` must have shape ``[tokens, 3, H, D]``
    and ``cu_seqlens`` must contain the cumulative token boundaries.
    """

    del max_seqlen
    if args:
        raise TypeError("Positional options beyond PTv3's API are unsupported")
    unsupported = {key: value for key, value in kwargs.items() if value is not None}
    if unsupported:
        raise TypeError(f"Unsupported flash_attn options: {sorted(unsupported)}")
    if qkv.ndim != 4 or qkv.shape[1] != 3:
        raise ValueError(f"qkv must have shape [tokens, 3, heads, dim], got {qkv.shape}")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a 1-D cumulative-length tensor")

    boundaries = cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
    if boundaries[0] != 0 or boundaries[-1] != qkv.shape[0]:
        raise ValueError(
            f"Invalid cu_seqlens endpoints {boundaries[0], boundaries[-1]} "
            f"for {qkv.shape[0]} tokens"
        )
    lengths = [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError(f"All sequence lengths must be positive, got {lengths}")

    if all(length == lengths[0] for length in lengths):
        return _uniform_varlen_attention(
            qkv,
            len(lengths),
            lengths[0],
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    return _variable_attention(
        qkv,
        boundaries,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )
