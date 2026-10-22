#!/usr/bin/env python3
"""Build Member handoff JSON + manifest previews from P3 provisional CSVs (no full train_manifest in git)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

TRAIN_COLS = [
    "sample_uid", "dataset", "dataset_root_id", "relative_path", "identity_key",
    "class_id", "raw_camera_or_view", "camera_semantics", "camera_context_id",
    "video_id", "tracklet_id", "frame_id", "clothes_id", "candidate_pool",
    "source_row_number",
]
VAL_COLS = [
    "sample_uid", "dataset", "dataset_root_id", "relative_path", "identity_key",
    "validation_protocol_id", "retrieval_role", "raw_camera_or_view",
    "camera_semantics", "camera_context_id", "video_id", "tracklet_id",
    "frame_id", "clothes_id", "candidate_pool", "source_row_number",
]
ROOT_ENV = {
    "ORBench_RGB": "DGREID_ROOT_ORBENCH",
    "PRID2011": "DGREID_ROOT_PRID2011",
    "RPIfield": "DGREID_ROOT_RPIFIELD",
    "CUHK03_NP": "DGREID_ROOT_CUHK03_NP",
    "DeepChange": "DGREID_ROOT_DEEPCHANGE",
    "Occluded-PoseTrack-ReID": "DGREID_ROOT_OCCLUDED_POSETRACK_REID",
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_p3(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def class_id_offset_after(prior_member_dir: Path | None) -> tuple[int, str | None]:
    if prior_member_dir is None:
        return 0, None
    path = prior_member_dir / "identity_class_mapping.json"
    if not path.is_file():
        raise SystemExit(f"Missing prior mapping: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping = data.get("identity_to_class", data)
    if not mapping:
        return 0, prior_member_dir.name
    return max(int(v) for v in mapping.values()) + 1, prior_member_dir.name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member-dir", type=Path, required=True)
    ap.add_argument("--p3-dir", type=Path, default=None)
    ap.add_argument("--preview-rows", type=int, default=10)
    ap.add_argument(
        "--after-member-dir",
        type=Path,
        default=None,
        help="Continue numeric class_id after this member's identity_class_mapping.json (e.g. Member 1)",
    )
    ap.add_argument(
        "--class-id-offset",
        type=int,
        default=None,
        help="Override first class_id (default: 0 or max(prior)+1)",
    )
    args = ap.parse_args()
    member = args.member_dir.resolve()
    p3_dir = (args.p3_dir or member / "p3").resolve()
    preview_dir = member / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    p3_files = sorted(p3_dir.glob("*_p3_provisional.csv"))
    if not p3_files:
        raise SystemExit(f"No *_p3_provisional.csv under {p3_dir}")

    split_report_path = p3_dir / "p3_split_preview.json"
    split_report = json.loads(split_report_path.read_text(encoding="utf-8")) if split_report_path.is_file() else {}

    train_keys: list[str] = []
    per_dataset: dict[str, dict] = {}
    train_preview: list[dict] = []
    val_preview: list[dict] = []
    source_versions: list[dict] = []

    for p3_path in p3_files:
        rows = read_p3(p3_path)
        if not rows:
            continue
        dataset = rows[0]["dataset"]
        train_rows = [r for r in rows if r.get("p3_role") == "train"]
        val_rows = [r for r in rows if r.get("p3_role") == "validation"]
        train_id_keys = sorted({r["identity_key"] for r in train_rows})
        train_keys.extend(train_id_keys)
        per_dataset[dataset] = {
            "train_rows": len(train_rows),
            "validation_rows": len(val_rows),
            "train_identities": len(train_id_keys),
            "validation_identities": len({r["identity_key"] for r in val_rows}),
        }
        unified_name = p3_path.name.replace("_p3_provisional.csv", "_unified_v1.csv")
        source_versions.append({
            "dataset": dataset,
            "filename": p3_path.name,
            "role": "p3_assignments",
            "sha256": file_sha256(p3_path),
        })
        info = (split_report.get("datasets") or {}).get(dataset, {})
        if info.get("input_sha256"):
            source_versions.append({
                "dataset": dataset,
                "filename": unified_name,
                "role": "frozen_unified",
                "sha256": info["input_sha256"],
            })

    train_keys = sorted(set(train_keys))
    auto_offset, prior_name = class_id_offset_after(
        args.after_member_dir.resolve() if args.after_member_dir else None
    )
    base = args.class_id_offset if args.class_id_offset is not None else auto_offset
    identity_to_class = {k: base + i for i, k in enumerate(train_keys)}

    for p3_path in p3_files:
        for r in read_p3(p3_path):
            role = r.get("p3_role")
            if role == "train" and len(train_preview) < args.preview_rows:
                train_preview.append({
                    **{c: r.get(c, "") for c in TRAIN_COLS if c != "class_id"},
                    "class_id": str(identity_to_class[r["identity_key"]]),
                })
            elif role == "validation" and len(val_preview) < args.preview_rows:
                val_preview.append({
                    **{c: r.get(c, "") for c in VAL_COLS if c not in ("validation_protocol_id", "retrieval_role")},
                    "validation_protocol_id": f"{r['dataset'].lower()}_provisional_identity_val_v1",
                    "retrieval_role": "query" if len(val_preview) % 2 == 0 else "gallery",
                })

    sources = []
    for dataset, stats in per_dataset.items():
        root_id = dataset
        sources.append({
            "dataset": dataset,
            "dataset_root_id": root_id,
            "relative_base_path": "",
            "root_env_var": ROOT_ENV.get(dataset, f"DGREID_ROOT_{dataset.replace('-', '_')}"),
            "train_rows": stats["train_rows"],
            "validation_unique_rows": stats["validation_rows"],
        })

    mapping_doc = {
        "schema_version": "v1",
        "class_id_offset": base,
        "continues_after_member": prior_name,
        "num_training_identities": len(train_keys),
        "class_id_range_inclusive": [base, base + len(train_keys) - 1] if train_keys else [base, base],
        "identity_to_class": identity_to_class,
    }
    (member / "identity_class_mapping.json").write_text(
        json.dumps(mapping_doc, indent=2) + "\n",
        encoding="utf-8",
    )
    (member / "source_registry.json").write_text(
        json.dumps({"schema_version": "v1", "sources": sources}, indent=2) + "\n",
        encoding="utf-8",
    )

    policies = {}
    for name, entry in (split_report.get("datasets") or {}).items():
        policies[name] = {
            "split_policy": entry.get("split_policy"),
            "split_meta": entry.get("split_meta"),
            "train_identities": entry.get("train_identities"),
            "validation_identities": entry.get("validation_identities"),
            "train_images": entry.get("train_images"),
            "validation_images": entry.get("validation_images"),
        }

    protocol = {
        "caveats": [
            "Metadata and P3 retrieval checks cannot prove source-to-source/target acquisition independence.",
            "Actual image bytes, runtime root paths, and pretrained checkpoint provenance were not checked.",
            "Full train_manifest.csv and large P3 CSVs are local-only (gitignored); previews + JSON in repo.",
        ],
        "data_images_and_paths_verified": False,
        "exclusion_policy": {
            "note": "No new exclusions invented or applied in P5; P4 must be completed separately",
            "p4_approval_evidence": None,
            "p4_status": "NOT_CLEARED",
        },
        "expected_full_program_sources": 9,
        "num_selected_sources": len(per_dataset),
        "p3_p4_externally_approved": False,
        "protocol_id": "dg_reid_multi_source_p5_v1",
        "ready_for_finetuning": False,
        "schema_version": "v1",
        "scope": "Only the selected datasets; NOT all nine unless all members merged to Global",
        "selected_datasets": sorted(per_dataset),
        "source_versions": source_versions,
        "split_policy": {
            "identity_disjointness_checked": True,
            "method": "P3 per-image roles from build_member_handoff_json / p3_split_sources",
            "policy_mode": split_report.get("policy_mode"),
            "seed_reported": split_report.get("seed"),
            "per_dataset": policies,
        },
        "status": "PROVISIONAL_P3_P4_PENDING",
        "training_sources": sorted(per_dataset),
        "global_class_mapping": {
            "class_id_offset": base,
            "continues_after_member": prior_name,
            "num_training_identities": len(train_keys),
            "class_id_range_inclusive": [base, base + len(train_keys) - 1] if train_keys else [base, base],
        },
    }
    (member / "dataset_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    train_prev = member / "preview" / "train_manifest_preview.csv"
    val_prev = member / "preview" / "validation_manifest_preview.csv"
    with train_prev.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRAIN_COLS)
        w.writeheader()
        w.writerows(train_preview)
    with val_prev.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=VAL_COLS)
        w.writeheader()
        w.writerows(val_preview)

    checksum_targets = [
        member / "identity_class_mapping.json",
        member / "source_registry.json",
        member / "dataset_protocol.json",
        p3_dir / "p3_split_preview.json",
        train_prev,
        val_prev,
    ]
    lines = []
    for p in checksum_targets:
        if p.is_file():
            lines.append(f"{file_sha256(p)}  {p.name if p.parent == member else p.relative_to(member).as_posix()}")
    (member / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote handoff under {member} ({len(train_keys)} training identities)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
