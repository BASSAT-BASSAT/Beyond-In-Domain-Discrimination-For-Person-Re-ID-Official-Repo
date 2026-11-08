#!/usr/bin/env python3
"""Deployment cost of the study backbones: parameters, MACs, latency, throughput.

Models are built from the same study conditions used for training (SBS ``R0``,
``OSNet_R0``, ``ConvNeXtS_R0``) at the test resolution, in inference mode, so the
identity classifier is not part of the measured forward pass. Objective choice
(e.g. ArcFace vs. Circle-softmax) does not change these numbers.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CONDITIONS = {
    "SBS R50-IBN": ("R0", False),
    "OSNet": ("OSNet_R0", True),
    "ConvNeXt-S": ("ConvNeXtS_R0", True),
}


def build(condition: str, backbone_study: bool, device: str):
    from reidfm.config import load_cfg
    from reidfm.experiments.loss_study import (
        expand_loss_study_run,
        load_backbone_study_manifest,
    )
    from reidfm.fastreid_compat import ensure_fastreid

    ensure_fastreid()
    from fastreid.modeling.meta_arch import build_model

    manifest = load_backbone_study_manifest() if backbone_study else None
    run = expand_loss_study_run(condition, 0, manifest) if manifest else expand_loss_study_run(condition, 0)
    cfg = load_cfg(run.config_file, run.opts)
    cfg.defrost()
    cfg.MODEL.DEVICE = device
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.HEADS.NUM_CLASSES = 10036
    cfg.freeze()
    model = build_model(cfg).to(device).eval()
    return model, tuple(cfg.INPUT.SIZE_TEST)


def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    classifier = 0
    for name, p in model.named_parameters():
        if name.startswith("heads.") and ("weight" == name.split(".")[-1]) and p.dim() == 2 and p.shape[0] == 10036:
            classifier += p.numel()
        elif name.startswith("heads.") and p.dim() >= 2 and p.shape[0] in (10036, 3 * 10036):
            classifier += p.numel()
    return total, total - classifier


def count_macs(model, size, device) -> float:
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    x = torch.zeros(1, 3, *size, device=device)
    with torch.inference_mode(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 2.0  # multiply-accumulates


def time_forward(model, size, device, batch, amp, warmup, iters) -> list[float]:
    import torch

    x = torch.randn(batch, 3, *size, device=device)
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if amp and device.startswith("cuda")
        else torch.autocast(device_type="cpu", enabled=False)
    )
    times = []
    with torch.inference_mode(), ctx:
        for _ in range(warmup):
            model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return times


def cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--throughput-batch", type=int, default=64)
    parser.add_argument("--cpu-batch", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--cpu-warmup", type=int, default=5)
    parser.add_argument("--cpu-iters", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--part", choices=("gpu", "cpu", "all"), default="all",
                        help="time only one device per process so heat from one pass does not bias the other")
    parser.add_argument("--out", default="reports/paper_figs/deployment_cost.json")
    args = parser.parse_args()

    import torch

    torch.backends.cudnn.benchmark = True
    out = REPO_ROOT / args.out
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"models": {}}
    results["environment"] = {
        "gpu": torch.cuda.get_device_name(0),
        "cpu": cpu_name(),
        "cpu_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "os": platform.platform(),
    }
    for label in args.models:
        condition, backbone_study = CONDITIONS[label]
        model, size = build(condition, backbone_study, "cuda")
        total, deployed = count_params(model)
        row = results["models"].setdefault(label, {})
        row.update({
            "condition": condition,
            "input": list(size),
            "params_total_M": total / 1e6,
            "params_deployed_M": deployed / 1e6,
            "gmacs": count_macs(model, size, "cuda") / 1e9,
        })

        def repeated(device, batch, amp, warmup, iters):
            # Median per repeat, then median (and range) across repeats.
            meds = [statistics.median(time_forward(model, size, device, batch, amp, warmup, iters))
                    for _ in range(args.repeats)]
            return statistics.median(meds), min(meds), max(meds)

        if args.part in ("gpu", "all"):
            lat32 = repeated("cuda", 1, False, args.warmup, args.iters)
            lat16 = repeated("cuda", 1, True, args.warmup, args.iters)
            thr = repeated("cuda", args.throughput_batch, True, 10, max(30, args.iters // 5))
            row.update({
                "gpu_latency_ms_fp32_b1": 1000 * lat32[0],
                "gpu_latency_ms_fp32_b1_range": [1000 * lat32[1], 1000 * lat32[2]],
                "gpu_latency_ms_fp16_b1": 1000 * lat16[0],
                "gpu_throughput_img_s_fp16": args.throughput_batch / thr[0],
                "gpu_throughput_img_s_fp16_range": [args.throughput_batch / thr[2], args.throughput_batch / thr[1]],
                "gpu_batch": args.throughput_batch,
            })
            print(f"{label:<12} GPU b1 fp32={row['gpu_latency_ms_fp32_b1']:.2f}ms "
                  f"thr={row['gpu_throughput_img_s_fp16']:.0f} img/s", flush=True)
        if args.part in ("cpu", "all"):
            model = model.cpu()
            torch.cuda.empty_cache()
            cpu_lat = repeated("cpu", 1, False, args.cpu_warmup, args.cpu_iters)
            cpu_thr = repeated("cpu", args.cpu_batch, False, 2, max(8, args.cpu_iters // 6))
            row.update({
                "cpu_latency_ms_fp32_b1": 1000 * cpu_lat[0],
                "cpu_latency_ms_fp32_b1_range": [1000 * cpu_lat[1], 1000 * cpu_lat[2]],
                "cpu_throughput_img_s_fp32": args.cpu_batch / cpu_thr[0],
                "cpu_throughput_img_s_fp32_range": [args.cpu_batch / cpu_thr[2], args.cpu_batch / cpu_thr[1]],
                "cpu_batch": args.cpu_batch,
            })
            print(f"{label:<12} CPU b1={row['cpu_latency_ms_fp32_b1']:.1f}ms "
                  f"thr={row['cpu_throughput_img_s_fp32']:.1f} img/s", flush=True)
        del model
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
