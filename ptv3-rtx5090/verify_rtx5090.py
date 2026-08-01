#!/usr/bin/env python3
"""Fail-fast GPU verification for the PTv3 RTX 5090 container."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch


def check_runtime() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Start the container with --gpus all and verify "
            "that NVIDIA Container Toolkit works on the host."
        )
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"GPU: {name}; compute capability: {capability}")
    print(f"PyTorch: {torch.__version__}; wheel CUDA: {torch.version.cuda}")
    if capability != (12, 0):
        raise RuntimeError(f"Expected RTX 50-series sm_120, found {capability}")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Expected the cu128 PyTorch wheel, found {torch.version.cuda}")
    if "sm_120" not in torch.cuda.get_arch_list():
        raise RuntimeError(
            f"This PyTorch wheel was not built for sm_120: {torch.cuda.get_arch_list()}"
        )


def check_fused_attention() -> None:
    import flash_attn

    if "ptv3.sdpa" not in flash_attn.__version__:
        raise RuntimeError(
            "The PTv3 fused-SDPA compatibility package was replaced: "
            f"flash_attn={flash_attn.__version__}"
        )

    batch, length, heads, head_dim = 4, 1024, 8, 64
    qkv = torch.randn(
        batch * length,
        3,
        heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    cu_seqlens = torch.arange(
        0,
        (batch + 1) * length,
        length,
        device="cuda",
        dtype=torch.int32,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = flash_attn.flash_attn_varlen_qkvpacked_func(
        qkv,
        cu_seqlens,
        max_seqlen=length,
        dropout_p=0.0,
    )
    out.square().mean().backward()
    torch.cuda.synchronize()
    if out.shape != (batch * length, heads, head_dim):
        raise RuntimeError(f"Unexpected attention output shape: {out.shape}")
    if qkv.grad is None or not torch.isfinite(qkv.grad).all():
        raise RuntimeError("Fused attention backward returned an invalid gradient")
    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    print(f"Fused SDPA forward/backward: OK; peak allocated: {peak_mib:.1f} MiB")


def unique_grid(side: int, *, channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    count = side**3
    linear = torch.arange(count, device="cuda", dtype=torch.int32)
    z = torch.div(linear, side * side, rounding_mode="floor")
    y = torch.div(linear, side, rounding_mode="floor").remainder(side)
    x = linear.remainder(side)
    batch = torch.zeros_like(linear)
    coords = torch.stack((batch, z, y, x), dim=1).contiguous()
    feats = torch.randn(count, channels, device="cuda", requires_grad=True)
    return feats, coords


def check_spconv(iterations: int) -> None:
    import spconv.pytorch as spconv

    side, channels = 16, 8
    conv = spconv.SubMConv3d(channels, 16, 3, indice_key="verify-sm120").cuda()
    for step in range(iterations):
        feats, coords = unique_grid(side, channels=channels)
        sparse = spconv.SparseConvTensor(
            feats,
            coords,
            spatial_shape=[side, side, side],
            batch_size=1,
        )
        out = conv(sparse)
        loss = out.features.square().mean()
        loss.backward()
        torch.cuda.synchronize()
        if feats.grad is None or not torch.isfinite(feats.grad).all():
            raise RuntimeError(f"spconv invalid gradient at iteration {step}")
        conv.zero_grad(set_to_none=True)
    print(f"spconv SubMConv3d forward/backward x{iterations}: OK")


def resolve_pointcept(root: Path | None):
    candidates = []
    if root is not None:
        candidates.append(root)
    candidates.append(Path("/opt/pointcept"))
    for candidate in candidates:
        if (candidate / "pointcept").is_dir():
            sys.path.insert(0, str(candidate))
            break
    importlib.invalidate_caches()
    return importlib.import_module(
        "pointcept.models.point_transformer_v3.point_transformer_v3m1_base"
    )


def check_pointcept(root: Path | None, iterations: int) -> None:
    module = resolve_pointcept(root)
    PointTransformerV3 = module.PointTransformerV3
    model = PointTransformerV3(
        in_channels=6,
        order=("z",),
        stride=(2,),
        enc_depths=(1, 1),
        enc_channels=(32, 64),
        enc_num_head=(4, 8),
        enc_patch_size=(128, 128),
        dec_depths=(1,),
        dec_channels=(32,),
        dec_num_head=(4,),
        dec_patch_size=(128,),
        drop_path=0.0,
        shuffle_orders=False,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
    ).cuda().train()

    side = 16
    count = side**3
    for step in range(iterations):
        feat, sparse_coords = unique_grid(side, channels=6)
        data = {
            "feat": feat,
            "coord": sparse_coords[:, 1:].to(dtype=torch.float32),
            "grid_coord": sparse_coords[:, 1:],
            "offset": torch.tensor([count], device="cuda", dtype=torch.int32),
        }
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            point = model(data)
            loss = point.feat.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        if feat.grad is None or not torch.isfinite(feat.grad).all():
            raise RuntimeError(f"PTv3 invalid gradient at iteration {step}")
        model.zero_grad(set_to_none=True)
    print(f"Pointcept PTv3 fused-attention forward/backward x{iterations}: OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointcept-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--skip-pointcept",
        action="store_true",
        help="Only test CUDA, fused attention, and spconv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    check_runtime()
    check_fused_attention()
    check_spconv(args.iterations)
    if not args.skip_pointcept:
        check_pointcept(args.pointcept_root, args.iterations)
    print("RTX5090_PTV3_ALL_CHECKS_OK")


if __name__ == "__main__":
    main()
