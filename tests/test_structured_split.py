"""Camera/scene → identity → dHash fallback splits."""

from __future__ import annotations

from pathlib import Path

from reidfm.data.manifest import Manifest, ManifestRow
from reidfm.data.structured_split import assign_pid_groups, camera_or_scene_groups, split_structured_or_dhash


def _manifest(rows: list[ManifestRow]) -> Manifest:
    return Manifest("demo", Path("/tmp"), rows)


def test_camera_groups_split_mot_style():
    rows = []
    for seq in ("MOT17-02-FRCNN", "MOT17-04-FRCNN"):
        for tid in (1, 2, 3):
            pid = f"{seq}-{tid}"
            rows.append(ManifestRow(f"{pid}.jpg", pid, seq, "mot17", "train"))
    m = _manifest(rows)
    groups = camera_or_scene_groups(m, "mot17")
    assert len(groups) >= 2
    train, val, meta = assign_pid_groups(groups, train_fraction=0.8, seed=0)
    assert set(train) & set(val) == set()
    assert len(val) >= 1
    assert meta["strategy"] == "camera_scene"
    assert 0.5 <= meta["train_fraction_actual"] <= 0.95


def test_fallback_identity_when_single_camera():
    rows = [ManifestRow(f"{i}.jpg", str(i), "0", "demo", "train") for i in range(10)]
    m = _manifest(rows)

    def _no_dhash(*_a, **_k):
        raise AssertionError("dhash should not run")

    train, val, _, meta = split_structured_or_dhash(
        "demo",
        m,
        train_fraction=0.8,
        max_samples_per_id=4,
        dhash_threshold=6,
        seed=0,
        rng=__import__("random").Random(0),
        dhash_fn=_no_dhash,
    )
    assert meta["strategy"] == "identity_disjoint"
    assert len(val) == 2
    assert len(train) == 8
