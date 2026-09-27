"""Embed images or tracklets with a registered model / checkpoint.

Input layouts:
  --images DIR      every image under DIR -> one row each
  --tracklets DIR   each sub-folder of DIR is a tracklet -> one mean-pooled, L2-normalized row
                    (linspace-sampled --num-frames frames, the evaluation protocol)

Output: an .npz with ``features`` [N, D] float32 (L2-normalized) and ``names`` [N].

Examples:
    python scripts/extract_embeddings.py --model reidfm_sbs_r50_ibn_v1 --images crops/ --out feats.npz
    python scripts/extract_embeddings.py --checkpoint outputs/run/model_best.pth --tracklets tracks/ --out t.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from reidfm.data.converters.base import list_images  # noqa: E402
from reidfm.fastreid_compat import ensure_fastreid  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model")
    source.add_argument("--checkpoint")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--images", type=Path)
    inputs.add_argument("--tracklets", type=Path)
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=None, help="image-decoding processes (default 2 on Windows, 4 elsewhere)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ensure_fastreid()
    from reidfm.evaluation.aggregate import mean_pool_l2
    from reidfm.evaluation.extractors import build_extractor
    from reidfm.evaluation.metrics import l2_normalize
    from reidfm.evaluation.sampling import sample_indices

    extractor, _ = build_extractor(model_key=args.model, checkpoint=args.checkpoint, config_file=args.config_file,
                                   device=args.device, batch_size=args.batch_size, workers=args.workers,
                                   amp=not args.no_amp)
    if args.images:
        paths = list_images(args.images, recursive=True)
        features = l2_normalize(extractor.extract(paths))
        names = [str(p.relative_to(args.images)) for p in paths]
    else:
        tracklets = sorted(p for p in args.tracklets.iterdir() if p.is_dir())
        groups = []
        for folder in tracklets:
            frames = sorted(list_images(folder))
            groups.append([frames[i] for i in sample_indices(len(frames), args.num_frames)] if frames else [])
        flat = [p for g in groups for p in g]
        frame_feats = extractor.extract(flat)
        features, names, offset = [], [], 0
        for folder, group in zip(tracklets, groups):
            if not group:
                continue
            features.append(mean_pool_l2(frame_feats[offset:offset + len(group)]))
            names.append(folder.name)
            offset += len(group)
        features = np.stack(features)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, features=features.astype(np.float32), names=np.array(names))
    print(f"{len(names)} embeddings of dim {features.shape[1]} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
