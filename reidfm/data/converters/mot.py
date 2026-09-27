"""MOT17 / MOT20 ground-truth person crops (train split only; test GT is not public).

Layout: ``<root>/train/<SEQ>[-<DET>]/{img1/*.jpg, gt/gt.txt}``. gt.txt columns:
``frame, id, x, y, w, h, consider, class, visibility``.

Kept boxes: ``consider == 1``, ``class`` in ``classes`` (default pedestrian = 1),
``visibility >= min_visibility`` and ``h >= min_box_height``. MOT17 ships every sequence
three times (DPM/FRCNN/SDP) with identical frames and GT, so only ``detector`` is used.
Identity = ``<SEQ>-<id>``, camera = sequence, tracklet = identity (thin with
``train_frame_stride`` / ``max_train_per_id``, frames are highly redundant).
"""

from __future__ import annotations

from pathlib import Path

from reidfm.data.converters.base import SourceRow, find_dir


def _sequences(root: Path, detector: str | None) -> list[Path]:
    train = root / "train" if (root / "train").is_dir() else find_dir(root, ["train"], max_depth=2)
    if train is None:
        raise FileNotFoundError(f"MOT: no train/ folder under {root}")
    seqs = sorted(p for p in train.iterdir() if (p / "gt" / "gt.txt").is_file())
    if detector:
        seqs = [p for p in seqs if p.name.endswith(f"-{detector}")] or seqs
    return seqs


def convert_mot(raw: Path, options: dict, default_detector: str | None) -> list[SourceRow]:
    classes = {int(c) for c in options.get("classes", [1])}
    min_vis = float(options.get("min_visibility", 0.3))
    min_height = float(options.get("min_box_height", 50))
    stride = max(1, int(options.get("frame_stride", 1)))
    rows: list[SourceRow] = []
    for seq_dir in _sequences(raw, options.get("detector", default_detector)):
        seq = seq_dir.name.rsplit("-", 1)[0] if default_detector else seq_dir.name
        images = seq_dir / "img1"
        for line in (seq_dir / "gt" / "gt.txt").read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            frame, track = int(float(parts[0])), int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            consider, cls, vis = int(float(parts[6])), int(float(parts[7])), float(parts[8])
            if consider != 1 or cls not in classes or vis < min_vis or h < min_height or frame % stride:
                continue
            pid = f"{seq}-{track}"
            rows.append(
                SourceRow(
                    src=images / f"{frame:06d}.jpg",
                    pid=pid,
                    camid=seq,
                    split="train",
                    tracklet_id=pid,
                    frame=frame,
                    box=(x, y, x + w, y + h),
                )
            )
    return rows


def convert_mot17(raw: Path, options: dict) -> list[SourceRow]:
    return convert_mot(raw, options, default_detector="FRCNN")


def convert_mot20(raw: Path, options: dict) -> list[SourceRow]:
    return convert_mot(raw, options, default_detector=None)
