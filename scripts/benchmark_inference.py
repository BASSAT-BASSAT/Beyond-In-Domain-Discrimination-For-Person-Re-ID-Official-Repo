#!/usr/bin/env python3
"""Benchmark SBS inference latency (FPS) and parameter count for paper Pareto tables."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sbs_r50_ibn_msmt", help="registry key")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    from reidfm.evaluation.extractors import build_extractor

    extractor, _ = build_extractor(
        model_key=args.model if not args.checkpoint else None,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
    )
    net = extractor.model
    params = sum(p.numel() for p in net.parameters())

    x = torch.randn(args.batch_size, 3, args.height, args.width, device=args.device)
    net.eval()
    with torch.inference_mode():
        for _ in range(args.warmup):
            net(x)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            net(x)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    images = args.iters * args.batch_size
    fps = images / elapsed
    print(f"model={args.model or args.checkpoint}")
    print(f"params={params:,}")
    print(f"input={args.height}x{args.width} batch={args.batch_size}")
    print(f"fps={fps:.1f}  ms/image={1000.0 * elapsed / images:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
