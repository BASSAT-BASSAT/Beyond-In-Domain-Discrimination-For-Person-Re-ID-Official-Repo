#!/usr/bin/env python3
"""Merge the three *verified member handoffs* into ONE provisional global P3/P5 bundle.

Inputs: unified/final_dataset/Member {1,2,3}/{six standard files}.
Output: unified/final_dataset/Global/{six standard files}.

- Exact train/validation schema, class mapping, identities, roots and hashes validated.
- Keeps existing global class IDs; NEVER resplits or relabels source rows.
- Preserves multiple legitimate CUHK03 retrieval protocols (one image can occur
  in different validation protocols, but never twice within a single protocol).
- Refuses to overwrite an existing Global handoff and never edits member files.
- No actual images accessed. P4 NOT CLEARED. Do not commit full manifest CSVs.

Python 3.10+; only standard library.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

MEMBERS = ("Member 1", "Member 2", "Member 3")
CLASS_COUNTS = (1774, 728, 7534)
EXPECTED_TOTAL_CLASSES = 10036
EXPECTED_TOTAL_UNIQUE_IMAGES = 940809 + 21379
EXPECTED_P2_CANDIDATE_IDENTITIES = 11297
EXPECTED_LAST_OFFICIAL_QUERY_IDS = 56
EXPECTED_LAST_DISTRACTORS = 20000
EXPECTED_DATASETS = {
    "CUHK03_NP", "DeepChange", "Occluded-PoseTrack-ReID",
    "ORBench_RGB", "PRID2011", "RPIfield",
    "IUST_PersonReID", "LaST", "LPW",
}
TRAIN_COLUMNS = [
    "sample_uid", "dataset", "dataset_root_id", "relative_path", "identity_key",
    "class_id", "raw_camera_or_view", "camera_semantics", "camera_context_id",
    "video_id", "tracklet_id", "frame_id", "clothes_id", "candidate_pool",
    "source_row_number",
]
VAL_COLUMNS = [
    "sample_uid", "dataset", "dataset_root_id", "relative_path", "identity_key",
    "validation_protocol_id", "retrieval_role", "raw_camera_or_view",
    "camera_semantics", "camera_context_id", "video_id", "tracklet_id",
    "frame_id", "clothes_id", "candidate_pool", "source_row_number",
]
OUTPUT_FILES = (
    "train_manifest.csv", "validation_manifest.csv", "identity_class_mapping.json",
    "source_registry.json", "dataset_protocol.json", "checksums.sha256",
)
CHECKSUM_REQUIRED = set(OUTPUT_FILES) - {"checksums.sha256"}


class GlobalBuildError(ValueError):
    pass


def check(condition, message):
    if not condition:
        raise GlobalBuildError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(2 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    check(path.is_file(), f"Missing required file: {path}")
    with path.open("r", encoding="utf-8-sig") as fp:
        return json.load(fp)


def write_json(path: Path, obj):
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        json.dump(obj, fp, indent=2, sort_keys=True, ensure_ascii=False)
        fp.write("\n")


def checked_relative(rel: str, label: str):
    check(bool(rel) and "\\" not in rel and not rel.startswith("/") and
          not re.match(r"^[a-zA-Z]:", rel) and
          ".." not in PurePosixPath(rel).parts and ":" not in rel,
          f"{label}: unsafe relative_path {rel!r}")


def verify_member_checksums(base: Path):
    checksum = base / "checksums.sha256"
    check(checksum.is_file(), f"Missing checksum file {checksum}")
    entries = {}
    for line in checksum.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        check("  " in line, f"Malformed checksum entry: {base}: {line!r}")
        sha, rel = line.split("  ", 1)
        check(re.fullmatch(r"[0-9a-f]{64}", sha) is not None,
              f"Invalid SHA256 in {checksum}: {rel}")
        check(rel not in entries, f"Repeated checksum entry in {checksum}: {rel}")
        p = PurePosixPath(rel)
        check(not p.is_absolute() and ".." not in p.parts and ":" not in rel and "\\" not in rel,
              f"Unsafe checksum path in {checksum}: {rel}")
        full = base.joinpath(*p.parts)
        check(full.is_file(), f"Checksum source absent: {full}")
        check(digest(full) == sha, f"Checksum mismatch: {full}")
        entries[rel] = sha
    check(CHECKSUM_REQUIRED <= set(entries),
          f"{base}: checksums missing standard files: {sorted(CHECKSUM_REQUIRED-set(entries))}")
    return entries


def iter_csv(path: Path, columns):
    with path.open("r", newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        check(reader.fieldnames == columns,
              f"{path}: CSV columns differ from agreed P5 schema. Got: {reader.fieldnames}")
        for line_no, row in enumerate(reader, 2):
            check(None not in row and all(x is not None for x in row.values()),
                  f"{path}:{line_no}: malformed CSV row")
            yield line_no, row


def validate_row(row, file_label, member, registry):
    ds = row["dataset"]
    check(ds in registry, f"{file_label}: unknown dataset {ds!r} for {member}")
    check(row["dataset_root_id"] == registry[ds]["dataset_root_id"],
          f"{file_label}: root_id mismatch for {ds}: row={row['dataset_root_id']!r} "
          f"registry={registry[ds]['dataset_root_id']!r}")
    rel = row["relative_path"]
    checked_relative(rel, file_label)
    check(row["sample_uid"] == hashlib.sha256((ds + "\0" + rel).encode("utf-8")).hexdigest(),
          f"{file_label}: sample_uid does not match dataset + relative_path ({ds}/{rel})")
    check(bool(row["identity_key"]), f"{file_label}: missing identity_key")


def check_retrieval_positives(query_rows, gallery_rows):
    """Validate known per-protocol positive rule without running model inference.

    Dedup image identity checks are elsewhere. The context rule applies to custom
    cross-camera/view protocols; LaST stays official same-video friendly.
    """
    known = defaultdict(list)
    for r in gallery_rows:
        known[r["identity_key"]].append(r)
    for q in query_rows:
        proto = q["validation_protocol_id"]
        positives = known[q["identity_key"]]
        if proto.startswith("cuhk03_internal_"):
            positives = [x for x in positives if x["raw_camera_or_view"] != q["raw_camera_or_view"]]
        elif proto == "deepchange_internal_cross_tracklet_v1":
            check(bool(q["tracklet_id"]), f"Missing DeepChange query tracklet: {q['sample_uid']}")
            positives = [x for x in positives if x["tracklet_id"] and x["tracklet_id"] != q["tracklet_id"]]
        elif proto == "posetrack_internal_same_or_cross_video_v1":
            positives = [x for x in positives if x["sample_uid"] != q["sample_uid"]]
        elif proto == "last_official_val_query_gallery_v1":
            positives = [x for x in positives if x["sample_uid"] != q["sample_uid"]]
        else:
            # The repaired Member 2 and internal Member 3 protocols use camera_context_id.
            check(q["camera_context_id"], f"{proto}: query camera/view context missing")
            positives = [x for x in positives
                         if x["camera_context_id"] and x["camera_context_id"] != q["camera_context_id"]]
        check(positives, f"{proto}: query {q['sample_uid']} has no valid gallery positive")


def build(repo: Path):
    target_parent = repo / "unified" / "final_dataset"
    output = target_parent / "Global"
    check(target_parent.is_dir(), f"Missing folder {target_parent}")
    check(not output.is_file(), f"Output is not a directory: {output}")
    if output.exists():
        extra = [p.name for p in output.iterdir() if p.name != ".gitkeep"]
        check(not extra, "Global already has files; refusing to overwrite existing Global handoff: "
              + ", ".join(extra))

    # First pass: validate every member's checksum-protected documents and the
    # *existing* globally contiguous class numbering, no reassignment.
    mappings, docs, roots, source_entries = {}, {}, {}, {}
    mapping_provenance = []
    expected_next = 0
    for member, expected_count in zip(MEMBERS, CLASS_COUNTS):
        base = target_parent / member
        check(base.is_dir(), f"Member handoff folder missing: {base}")
        for name in OUTPUT_FILES:
            check((base / name).is_file(), f"Member handoff is incomplete: {base / name}")
        verified_entries = verify_member_checksums(base)
        mapping_json = load_json(base / "identity_class_mapping.json")
        mp = mapping_json.get("identity_to_class", mapping_json)
        check(isinstance(mp, dict) and len(mp) == expected_count,
              f"{member}: expected {expected_count} class mappings; found {len(mp) if isinstance(mp,dict) else 'not a dict'}")
        check(all(isinstance(v, int) and not isinstance(v,bool) for v in mp.values()),
              f"{member}: invalid class_id values")
        check(set(mp.values()) == set(range(expected_next, expected_next + expected_count)),
              f"{member}: class IDs must cover {expected_next}..{expected_next+expected_count-1} exactly")
        check(not (set(mp) & set(mappings)), f"{member}: overlapping training identity keys across members")
        mappings.update(mp)
        expected_next += expected_count
        protocol = load_json(base / "dataset_protocol.json")
        registry = load_json(base / "source_registry.json")
        check(registry.get("schema_version") == "v1" and isinstance(registry.get("sources"), list),
              f"{member}: malformed source registry")
        member_roots = {}
        for s in registry["sources"]:
            dataset = s.get("dataset")
            check(dataset and dataset not in roots and dataset not in member_roots,
                  f"Duplicate or invalid dataset in registries: {dataset}")
            check(s.get("dataset_root_id") and s.get("root_env_var"),
                  f"{member}/{dataset}: missing root or env var")
            check(s.get("relative_base_path", "") == "",
                  f"{member}/{dataset}: unsupported nonempty relative_base_path")
            member_roots[dataset] = s
            roots[dataset] = s
            source_entries[dataset] = member
        check(set(protocol.get("selected_datasets", [])) == set(member_roots),
              f"{member}: selected_datasets differs from source_registry")
        check(protocol.get("ready_for_finetuning") is False,
              f"{member}: unexpected training approval; check protocol manually")
        check(protocol.get("status", "").startswith("PROVISIONAL"),
              f"{member}: expected provisional protocol status")
        docs[member] = {"mapping": mp, "protocol": protocol, "registry": member_roots, "dir": base}
        mapping_provenance.append({"member": member,
                                   "mapping_sha256": verified_entries["identity_class_mapping.json"],
                                   "train_manifest_sha256": verified_entries["train_manifest.csv"],
                                   "validation_manifest_sha256": verified_entries["validation_manifest.csv"],
                                   "dataset_protocol_sha256": verified_entries["dataset_protocol.json"],
                                   "source_registry_sha256": verified_entries["source_registry.json"]})
        print(f"[PASS] {member}: {expected_count:,} training classes; source files/checksums validated", flush=True)
    check(expected_next == EXPECTED_TOTAL_CLASSES and len(mappings) == EXPECTED_TOTAL_CLASSES,
          "Global training class count inconsistent: expected 10,036")
    check(set(roots) == EXPECTED_DATASETS,
          f"Expected 9 unique source datasets: missing={sorted(EXPECTED_DATASETS-set(roots))}, "
          f"extra={sorted(set(roots)-EXPECTED_DATASETS)}")
    m3_provenance = docs["Member 3"]["protocol"].get("prior_member_mappings", [])
    for old in ("Member 1", "Member 2"):
        refs = [x for x in m3_provenance if x.get("member") == old]
        check(len(refs) == 1 and refs[0].get("sha256") == digest(docs[old]["dir"] / "identity_class_mapping.json"),
              f"Member 3's class mapping provenance does not match the CURRENT {old} mapping")

    # Global file writing happens in a temporary staging folder. No member
    # file is ever opened for writing or changed.
    staging = Path(tempfile.mkdtemp(prefix=".global_stage_", dir=target_parent))
    try:
        train_uid_to_key = {}; val_uid_to_info = {}
        member_class_coverage = {name: set() for name in MEMBERS}
        member_train_ids = {name: set() for name in MEMBERS}
        member_val_ids = {name: set() for name in MEMBERS}
        source_counts = defaultdict(Counter)
        member_counts = defaultdict(Counter)
        validation_protocols = defaultdict(lambda: {"query": [], "gallery": []})
        seen_validation_protocol_uid = set()
        
        with (staging / "train_manifest.csv").open("w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=TRAIN_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for member in MEMBERS:
                for lineno, row in iter_csv(docs[member]["dir"] / "train_manifest.csv", TRAIN_COLUMNS):
                    where = f"{member}/train_manifest.csv:{lineno}"
                    validate_row(row, where, member, docs[member]["registry"])
                    key = row["identity_key"]
                    check(key in docs[member]["mapping"], f"{where}: identity not in member class mapping: {key}")
                    try:
                        class_id = int(row["class_id"])
                    except ValueError as error:
                        raise GlobalBuildError(f"{where}: invalid class_id") from error
                    check(class_id == docs[member]["mapping"][key], f"{where}: wrong class_id for {key}")
                    uid = row["sample_uid"]
                    check(uid not in train_uid_to_key, f"{where}: duplicate training sample_uid {uid}")
                    train_uid_to_key[uid] = key
                    member_class_coverage[member].add(class_id)
                    member_train_ids[member].add(key)
                    source_counts[row["dataset"]]["train"] += 1
                    member_counts[member]["train"] += 1
                    writer.writerow(row)
        for member in MEMBERS:
            check(member_train_ids[member] == set(docs[member]["mapping"]),
                  f"{member}: manifest training ID set differs from its class mapping")

        with (staging / "validation_manifest.csv").open("w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=VAL_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for member in MEMBERS:
                for lineno, row in iter_csv(docs[member]["dir"] / "validation_manifest.csv", VAL_COLUMNS):
                    where = f"{member}/validation_manifest.csv:{lineno}"
                    validate_row(row, where, member, docs[member]["registry"])
                    key = row["identity_key"]; uid = row["sample_uid"]
                    check(key not in mappings, f"{where}: validation identity is used for supervised TRAIN: {key}")
                    check(uid not in train_uid_to_key,
                          f"{where}: same sample in training and validation: {uid}")
                    if uid in val_uid_to_info:
                        old = val_uid_to_info[uid]
                        check(old == (row["dataset"], key, row["relative_path"], row["dataset_root_id"]),
                              f"{where}: repeated validation UID has inconsistent metadata")
                    else:
                        val_uid_to_info[uid] = (row["dataset"], key, row["relative_path"], row["dataset_root_id"])
                        source_counts[row["dataset"]]["validation_unique"] += 1
                        member_counts[member]["validation_unique"] += 1
                    proto = row["validation_protocol_id"]; role = row["retrieval_role"]
                    check(proto and role in ("query", "gallery"),
                          f"{where}: invalid validation_protocol_id or retrieval_role")
                    pair = (proto, uid)
                    check(pair not in seen_validation_protocol_uid,
                          f"{where}: duplicate image within same retrieval protocol {proto}/{uid}")
                    seen_validation_protocol_uid.add(pair)
                    check(proto not in validation_protocols or
                          (not validation_protocols[proto]["query"] and not validation_protocols[proto]["gallery"])
                          or validation_protocols[proto].get("dataset", row["dataset"]) == row["dataset"],
                          f"{where}: retrieval protocol mixes distinct datasets: {proto}")
                    validation_protocols[proto]["dataset"] = row["dataset"]
                    validation_protocols[proto][role].append(row)
                    member_val_ids[member].add(key)
                    source_counts[row["dataset"]]["validation_rows"] += 1
                    source_counts[row["dataset"]][role] += 1
                    member_counts[member]["validation_rows"] += 1
                    writer.writerow(row)
        check(not (set(mappings) & set().union(*member_val_ids.values())),
              "Global train/validation identity leakage")
        for proto, group in validation_protocols.items():
            qs = group["query"]; gs = group["gallery"]
            check(qs and gs, f"Retrieval protocol must have both queries and gallery: {proto}")
            check_retrieval_positives(qs, gs)
            print(f"[PASS] Validation {proto}: {len(qs):,} queries, {len(gs):,} gallery", flush=True)

        for ds, reg in roots.items():
            check(source_counts[ds]["train"] == int(reg["train_rows"]),
                  f"{ds}: train rows disagree with source_registry")
            check(source_counts[ds]["validation_unique"] == int(reg["validation_unique_rows"]),
                  f"{ds}: unique validation rows disagree with source_registry; "
                  f"got {source_counts[ds]['validation_unique']:,}, "
                  f"expected {reg['validation_unique_rows']}")
        # Candidate pools count unique P2 images. LaST official validation was
        # appended from the preexisting all-splits LaST audit manifest.
        n_unique = len(train_uid_to_key) + len(val_uid_to_info)
        check(n_unique == EXPECTED_TOTAL_UNIQUE_IMAGES,
              f"Unexpected unique image count: {n_unique:,}, expected {EXPECTED_TOTAL_UNIQUE_IMAGES:,} "
              "(940,809 clean candidates + 21,379 LaST official val)")

        # Candidate-pool held-out identities exclude LaST official validation,
        # which is not part of the frozen 11,297 training candidate identities.
        source_val_keys = defaultdict(set)
        last_query_keys = set()
        last_distractor_count = 0
        for proto, g in validation_protocols.items():
            ds = g["dataset"]
            if ds == "LaST":
                for q in g["query"]:
                    last_query_keys.add(q["identity_key"])
                for gallery in g["gallery"]:
                    if "::distractor::" in gallery["identity_key"]:
                        last_distractor_count += 1
            else:
                for r in g["query"] + g["gallery"]:
                    source_val_keys[ds].add(r["identity_key"])
        candidate_val_ids = sum(len(ids) for ids in source_val_keys.values())
        check(candidate_val_ids == EXPECTED_P2_CANDIDATE_IDENTITIES - EXPECTED_TOTAL_CLASSES,
              f"Held-out source identity count is {candidate_val_ids}, expected 1,261")
        check(len(last_query_keys) == EXPECTED_LAST_OFFICIAL_QUERY_IDS and last_distractor_count == EXPECTED_LAST_DISTRACTORS,
              f"LaST official validation metadata unexpected: query IDs={len(last_query_keys)}, "
              f"distractors={last_distractor_count}")

        source_stats = []
        for ds in sorted(roots):
            entry = dict(roots[ds])
            entry["train_rows"] = source_counts[ds]["train"]
            entry["validation_unique_rows"] = source_counts[ds]["validation_unique"]
            source_stats.append(entry)
        write_json(staging / "identity_class_mapping.json", {
            "schema_version": "v1", "scope": "Global nine-source provisional training pool",
            "num_classes": len(mappings), "class_id_range_inclusive": [0, len(mappings)-1],
            "identity_to_class": dict(sorted(mappings.items())),
        })
        write_json(staging / "source_registry.json", {
            "schema_version": "v1", "sources": source_stats,
        })
        write_json(staging / "dataset_protocol.json", {
            "schema_version": "v1", "protocol_id": "dg_reid_global_p3_p5_v1",
            "status": "PROVISIONAL_P4_PENDING", "ready_for_finetuning": False,
            "p4_status": "NOT_CLEARED", "p4_approval_evidence": None,
            "source_official_test_used": False,
            "source_images_or_paths_opened_in_this_run": False,
            "selected_datasets": sorted(roots), "num_selected_sources": len(roots),
            "candidate_images_p2": 940809, "candidate_identity_labels_p2": 11297,
            "last_official_validation_images_outside_p2_candidate_pool": 21379,
            "train_images": sum(v["train"] for v in source_counts.values()),
            "validation_manifest_rows": sum(v["validation_rows"] for v in source_counts.values()),
            "validation_unique_images": len(val_uid_to_info),
            "train_classes": len(mappings), "class_id_range_inclusive": [0,len(mappings)-1],
            "source_candidate_val_identities_p3": candidate_val_ids,
            "last_official_val_query_identities": len(last_query_keys),
            "last_official_val_gallery_distractors": last_distractor_count,
            "dataset_to_member": dict(sorted(source_entries.items())),
            "member_provenance": mapping_provenance,
            "member_counts": {m: dict(member_counts[m]) for m in MEMBERS},
            "per_dataset": {ds: dict(source_counts[ds]) for ds in sorted(roots)},
            "retrieval_protocols": {p: dict(dataset=g["dataset"],
                query_images=len(g["query"]),gallery_images=len(g["gallery"]))
                for p,g in sorted(validation_protocols.items())},
            "caveats": [
                "P4 cross-source/cross-target acquisition and image leakage NOT cleared.",
                "No real images opened or paths checked; verify all image roots on Kaggle before training.",
                "Source data is merged without revising the three member split policies or class IDs.",
                "Different internal validation protocols are not interchangeable with official benchmark protocols.",
                "LaST official validation contains 20,000 unlabeled gallery distractors, preserved as nonmatching IDs.",
                "Some Member 1 validation samples occur in two directional retrieval protocols; validate by (protocol, sample_uid).",
                "Market1501 and MARS remain unseen targets; do not use them for model selection.",
                "Keep both global full manifest CSVs local; push only previews and small metadata after review.",
            ],
        })
        with (staging / "checksums.sha256").open("w", encoding="utf-8", newline="\n") as fp:
            for name in OUTPUT_FILES[:-1]:
                fp.write(f"{digest(staging / name)}  {name}\n")
        # Verify every generated output's self-check before writing to Global.
        for name in OUTPUT_FILES[:-1]:
            entry = next(line for line in (staging / "checksums.sha256").read_text(encoding="utf-8").splitlines()
                         if line.endswith("  " + name))
            check(entry[:64] == digest(staging/name), f"Generated checksum verification failed: {name}")
        output.mkdir(exist_ok=True)
        for name in OUTPUT_FILES:
            (staging / name).replace(output / name)
        (output / ".gitkeep").unlink(missing_ok=True)
        print("\n[SUCCESS] Global six-file provisional handoff created:", flush=True)
        for name in OUTPUT_FILES:
            print("   " + str(output / name), flush=True)
        print(f"[TOTAL] Train: {sum(v['train'] for v in source_counts.values()):,} | "
              f"Validation rows: {sum(v['validation_rows'] for v in source_counts.values()):,} | "
              f"Unique images: {n_unique:,} | Training classes: {len(mappings):,}", flush=True)
        print("[STATUS] PROVISIONAL; P4 NOT CLEARED. Full CSVs must stay LOCAL.", flush=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2],
                    help="Repository root. Defaults to scripts/dg_reid_data_pipeline/../../")
    args = ap.parse_args()
    try:
        build(args.repo.resolve())
    except (GlobalBuildError, KeyError, OSError, json.JSONDecodeError, csv.Error, ValueError) as error:
        ap.exit(1, f"\n[STOP] {error}\nGlobal handoff was not approved. Member files were not changed.\n")


if __name__ == "__main__":
    main()
