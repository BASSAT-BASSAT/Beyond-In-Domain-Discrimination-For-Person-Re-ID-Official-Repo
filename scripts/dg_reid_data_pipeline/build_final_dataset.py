#!/usr/bin/env python3
"""P5 manifest builder for DG-ReID; no new splitting or leakage inference.

Produces exactly:
 final_dataset/v1/{train_manifest.csv,validation_manifest.csv,
 identity_class_mapping.json,source_registry.json,dataset_protocol.json,checksums.sha256}

Inputs: frozen Unified v1 CSVs, complete P3 role-assignment CSVs, and P3 retrieval CSVs.
Without external P3/P4 approvals it generates an explicitly PROVISIONAL bundle,
NOT a leakage-cleared dataset ready for supervised training.

Python 3.10+, standard library only. Doesn't modify source files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sqlite3
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

BASE_COLS = [
    "sample_uid", "dataset", "candidate_pool", "dataset_root_id", "relative_path",
    "source_split", "raw_pid", "identity_namespace", "identity_key",
    "raw_camera_or_view", "camera_semantics", "camera_context_id", "video_id",
    "tracklet_id", "frame_id", "clothes_id", "source_row_number",
    "source_image_index", "source_metadata_json",
]
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
# Canonical names come from the Unified v1 handoff; never derive them from raw PID.
SOURCES = {
    "cuhk03": ("CUHK03_NP", "cuhk03_np"),
    "iust": ("IUST_PersonReID", "iust"),
    "last": ("LaST", "last"),
    "deepchange": ("DeepChange", "deepchange"),
    "lpw": ("LPW", "lpw"),
    "occluded_posetrack": ("Occluded-PoseTrack-ReID", "occluded_posetrack"),
    "orbench": ("ORBench", "orbench_rgb"),
    "prid2011": ("PRID2011", "prid2011"),
    "rpifield": ("RPIfield", "rpifield"),
}
# Accepted observed retrieval protocols (not claimed to be official benchmarks).
PROTOCOL_DATASETS = {
    "cuhk03_internal_c1_to_c2_v1": "CUHK03_NP",
    "cuhk03_internal_c2_to_c1_v1": "CUHK03_NP",
    "deepchange_internal_cross_tracklet_v1": "DeepChange",
    "posetrack_internal_same_or_cross_video_v1": "Occluded-PoseTrack-ReID",
}
EXPECTED_P3_REPORT = "p3_final_verification_report.json"
EXPECTED_PROTOCOL_REPORT = "p3_retrieval_protocol_report.json"
OUTPUT_NAMES = [
    "train_manifest.csv", "validation_manifest.csv", "identity_class_mapping.json",
    "source_registry.json", "dataset_protocol.json", "checksums.sha256",
]

class DataError(RuntimeError):
    pass

def require(condition, message):
    if not condition:
        raise DataError(message)

def file_sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path, data):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

def csv_iter(handle, label, fields):
    reader = csv.DictReader(handle)
    require(reader.fieldnames is not None, f"Empty CSV: {label}")
    require(len(reader.fieldnames) == len(set(reader.fieldnames)), f"Duplicate header: {label}")
    absent = set(fields) - set(reader.fieldnames)
    require(not absent, f"{label}: Missing required columns: {sorted(absent)}")
    for lineno, row in enumerate(reader, 2):
        require(None not in row and all(v is not None for v in row.values()),
                f"{label}:{lineno}: Malformed CSV row")
        yield row

def locate_one(folder: Path, filename: str):
    found = sorted(p for p in folder.rglob(filename) if p.is_file())
    require(len(found) == 1, f"Expected ONE {filename} under {folder}; found {len(found)}")
    return found[0]

class RetrievalInputs:
    """Read either an extracted retrieval directory or a ZIP with the same content."""
    def __init__(self, directory=None, archive=None):
        self.directory = directory
        self.archive = zipfile.ZipFile(archive) if archive else None
        if self.archive:
            names = self.archive.namelist()
            self.csv_names = sorted(n for n in names if n.lower().endswith(".csv"))
            self.json_names = sorted(n for n in names if n.lower().endswith(".json"))
            self._zip_path = archive
        else:
            self.csv_names = sorted(str(p) for p in directory.rglob("*.csv"))
            self.json_names = sorted(str(p) for p in directory.rglob("*.json"))
            self._zip_path = None

    def open_text(self, filename):
        if self.archive:
            return io.TextIOWrapper(self.archive.open(filename), encoding="utf-8-sig", newline="")
        return open(filename, "r", encoding="utf-8-sig", newline="")

    def exact_one(self, basename, suffix="csv"):
        files = self.csv_names if suffix == "csv" else self.json_names
        found = [n for n in files if Path(n).name == basename]
        require(len(found) == 1, f"Expected ONE {basename} in retrieval; found {len(found)}")
        return found[0]

    def sha256(self, filename):
        if self.archive:
            h = hashlib.sha256()
            with self.archive.open(filename) as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
            return h.hexdigest()
        return file_sha256(Path(filename))

    def close(self):
        if self.archive:
            self.archive.close()

def validate_path(row):
    path = row["relative_path"]
    require(path and not path.startswith(("/", "\\")) and "\\" not in path and
            ".." not in path.split("/") and not re.match(r"^[A-Za-z]:", path),
            f"Unsafe relative path for UID {row['sample_uid']}: {path}")
    require(row["dataset_root_id"], f"Missing dataset_root_id: {row['sample_uid']}")
    expected = hashlib.sha256((row["dataset"] + "\x00" + path).encode("utf-8")).hexdigest()
    require(row["sample_uid"] == expected, f"sample_uid SHA mismatch: {row['sample_uid']}")
    require(row["identity_key"] == row["identity_namespace"] + "::" + row["raw_pid"],
            f"Bad identity namespace: {row['sample_uid']}")

def validate_positive_pairs(pid, queries, gallery):
    byid = defaultdict(list)
    for g in gallery:
        byid[g["identity_key"]].append(g)
    for q in queries:
        positives = byid[q["identity_key"]]
        if pid.startswith("cuhk03_"):
            positives = [g for g in positives
                         if g["raw_camera_or_view"] != q["raw_camera_or_view"]]
        elif pid == "deepchange_internal_cross_tracklet_v1":
            require(bool(q["tracklet_id"]), "DeepChange query has no tracklet_id")
            positives = [g for g in positives if g["tracklet_id"] and
                         g["tracklet_id"] != q["tracklet_id"]]
        elif pid == "posetrack_internal_same_or_cross_video_v1":
            positives = [g for g in positives if g["sample_uid"] != q["sample_uid"]]
        else:
            raise DataError(f"Unknown retrieval matching rule for protocol: {pid}")
        require(positives, f"No valid positive: protocol={pid}, query={q['sample_uid']}")


def build(args):
    selected = list(SOURCES) if args.datasets == ["all"] else args.datasets
    require(selected and len(selected) == len(set(selected)) and all(s in SOURCES for s in selected),
            "Unknown/duplicate dataset key; choose among: " + ", ".join(SOURCES))
    unified_dir, p3_dir = args.unified_dir.resolve(), args.p3_dir.resolve()
    require(unified_dir.is_dir(), f"Unified directory missing: {unified_dir}")
    require(p3_dir.is_dir(), f"P3 directory missing: {p3_dir}")
    require(args.retrieval_dir or args.retrieval_zip, "Provide --retrieval-dir OR --retrieval-zip")
    output = args.output_dir.resolve()
    require(not output.exists(), f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    retrieval = RetrievalInputs(directory=args.retrieval_dir, archive=args.retrieval_zip)
    try:
        report_name = retrieval.exact_one(EXPECTED_P3_REPORT, "json")
        rules_name = retrieval.exact_one(EXPECTED_PROTOCOL_REPORT, "json")
        with retrieval.open_text(report_name) as f: verify_report = json.load(f)
        with retrieval.open_text(rules_name) as f: protocol_report = json.load(f)
        require(verify_report.get("overall") == "PASS", "P3 retrieval report is not PASS")
        require(protocol_report.get("scope", "").lower().startswith("internal validation"),
                "Unexpected P3 report scope")

        with tempfile.TemporaryDirectory(prefix=".p5_staging_", dir=output.parent) as tmp:
            stage = Path(tmp)
            db = sqlite3.connect(str(stage / "_p5_build.sqlite"))
            try:
                db.execute("PRAGMA journal_mode=OFF")
                db.execute("PRAGMA synchronous=OFF")
                db.execute("CREATE TABLE p3 (uid TEXT PRIMARY KEY, dataset TEXT, identity_key TEXT, role TEXT, extra TEXT)")
                db.execute("CREATE TABLE samples (uid TEXT PRIMARY KEY, dataset TEXT, identity_key TEXT, role TEXT, data TEXT)")
                db.execute("CREATE TABLE retrieval (protocol TEXT, uid TEXT, role TEXT, PRIMARY KEY (protocol,uid,role))")
                db.execute("CREATE INDEX retrieval_uid_idx ON retrieval(uid)")
                db.execute("CREATE INDEX samples_dataset_role_idx ON samples(dataset,role)")
                inputs = []
                source_counts = {}
                source_roots = defaultdict(set)
    
                # A) Require COMPLETE per-image P3 assignments. Retrieval alone cannot derive train.
                for source in selected:
                    dataset, stem = SOURCES[source]
                    p3_name = f"{stem}_p3_provisional.csv"
                    p3_path = locate_one(p3_dir, p3_name)
                    uni_path = locate_one(unified_dir, f"{stem}_unified_v1.csv")
                    print(f"[P3] {dataset}: reading role decisions")
                    n_p3 = Counter()
                    with p3_path.open("r", encoding="utf-8-sig", newline="") as f:
                        for row in csv_iter(f, str(p3_path), ("sample_uid", "p3_role")):
                            uid = row["sample_uid"]
                            role = row["p3_role"].strip().lower()
                            require(role in ("train", "validation"),
                                    f"{p3_path.name}: unknown p3_role for {uid}: {role!r}")
                            if row.get("dataset"):
                                require(row["dataset"] == dataset, f"Wrong dataset in P3 {uid}")
                            try:
                                db.execute("INSERT INTO p3 VALUES (?,?,?,?,?)",
                                           (uid, dataset, row.get("identity_key", ""), role,
                                            json.dumps({k: row[k] for k in BASE_COLS if k in row})))
                            except sqlite3.IntegrityError as exc:
                                raise DataError(f"P3 duplicate sample_uid: {uid}") from exc
                            n_p3[role] += 1
                    require(sum(n_p3.values()) > 0, f"Empty P3 assignments for {dataset}")
                    print(f"[UNIFIED] {dataset}: checking identity/path/role lineage")
                    n_unified = Counter()
                    with uni_path.open("r", encoding="utf-8-sig", newline="") as f:
                        for row in csv_iter(f, str(uni_path), BASE_COLS):
                            require(row["dataset"] == dataset, f"Unified dataset mismatch: {row['sample_uid']}")
                            validate_path(row)
                            uid = row["sample_uid"]
                            found = db.execute("SELECT identity_key,role,extra FROM p3 WHERE uid=?", (uid,)).fetchone()
                            require(found is not None, f"P3 missing decision for Unified UID {uid}")
                            p3_identity, role, p3_fields_json = found
                            require(not p3_identity or p3_identity == row["identity_key"],
                                    f"P3 identity disagrees with Unified for UID {uid}")
                            for field, value in json.loads(p3_fields_json).items():
                                require(row[field] == value,
                                        f"P3 {field} differs from frozen Unified for UID {uid}")
                            try:
                                db.execute("INSERT INTO samples VALUES (?,?,?,?,?)",
                                           (uid, dataset, row["identity_key"], role, json.dumps(row)))
                            except sqlite3.IntegrityError as exc:
                                raise DataError(f"Duplicate Unified sample UID: {uid}") from exc
                            source_roots[dataset].add(row["dataset_root_id"])
                            n_unified[role] += 1
                    require(n_p3 == n_unified,
                            f"Not all Unified rows have exactly one P3 role for {dataset}: "
                            f"p3={dict(n_p3)} unified={dict(n_unified)}")
                    require(len(source_roots[dataset]) == 1,
                            f"Multiple dataset_root_id for {dataset}; needs explicit registry handling")
                    source_counts[dataset] = dict(n_unified)
                    inputs.extend([
                        {"role": "frozen_unified", "dataset": dataset, "filename": uni_path.name,
                         "sha256": file_sha256(uni_path)},
                        {"role": "p3_assignments", "dataset": dataset, "filename": p3_path.name,
                         "sha256": file_sha256(p3_path)},
                    ])
                    db.commit()
    
                bad = db.execute("SELECT identity_key FROM samples GROUP BY identity_key "
                                 "HAVING COUNT(DISTINCT role)>1 LIMIT 1").fetchone()
                require(bad is None, f"Train/val identity leakage within selected sources: {bad}")
                training_ids = [r[0] for r in db.execute(
                    "SELECT DISTINCT identity_key FROM samples WHERE role='train' ORDER BY identity_key")]
                require(training_ids, "No training identities in approved P3 role files")
                identity_to_class = {key: idx for idx, key in enumerate(training_ids)}
    
                # B) Validate retrieval roles against ALL P3 decisions and frozen Unified metadata.
                retrieval_csvs = [x for x in retrieval.csv_names if Path(x).name not in OUTPUT_NAMES]
                require(retrieval_csvs, "No retrieval CSVs provided")
                seen_protocols = set()
                protocol_rows = defaultdict(lambda: defaultdict(list))
                protocol_stats = {}
                for name in retrieval_csvs:
                    count = 0
                    local_protocol = None
                    with retrieval.open_text(name) as f:
                        for row in csv_iter(f, name, BASE_COLS + ["p3_role", "validation_protocol", "retrieval_role"]):
                            uid = row["sample_uid"]
                            protocol = row["validation_protocol"]
                            role = row["retrieval_role"]
                            require(protocol in PROTOCOL_DATASETS,
                                    f"Unknown validation_protocol; add and review explicit rules first: {protocol}")
                            require(row["dataset"] == PROTOCOL_DATASETS[protocol],
                                    f"Dataset/protocol mismatch: {protocol}")
                            require(row["dataset"] in source_counts,
                                    f"Retrieval uses dataset not selected: {row['dataset']}")
                            require(role in ("query", "gallery") and row["p3_role"] == "validation",
                                    f"Bad retrieval/p3_role for {uid}")
                            require(local_protocol in (None, protocol), f"Multiple protocols mixed within {name}")
                            local_protocol = protocol
                            saved = db.execute("SELECT role,data FROM samples WHERE uid=?", (uid,)).fetchone()
                            require(saved is not None and saved[0] == "validation",
                                    f"Retrieval UID not in P3 validation: {uid}")
                            original = json.loads(saved[1])
                            for key in BASE_COLS:
                                require(row[key] == original[key],
                                        f"Retrieval metadata changed from frozen Unified, {uid}, field={key}")
                            try:
                                db.execute("INSERT INTO retrieval VALUES (?,?,?)", (protocol, uid, role))
                            except sqlite3.IntegrityError as exc:
                                raise DataError(f"Duplicate protocol/sample/role row: {protocol}/{uid}/{role}") from exc
                            protocol_rows[protocol][role].append(row)
                            count += 1
                    require(count > 0, f"Empty retrieval file: {name}")
                    require(local_protocol not in seen_protocols, f"Protocol stored in multiple files: {local_protocol}")
                    require(Path(name).stem == local_protocol,
                            f"Retrieval filename does not match protocol ID: {name}")
                    seen_protocols.add(local_protocol)
                    inputs.append({"role": "p3_retrieval", "filename": Path(name).name,
                                   "sha256": retrieval.sha256(name)})
                    db.commit()
    
                for proto, roles in protocol_rows.items():
                    queries, gallery = roles.get("query", []), roles.get("gallery", [])
                    require(queries and gallery, f"Protocol missing query or gallery: {proto}")
                    validate_positive_pairs(proto, queries, gallery)
                    expected = verify_report.get("protocols", {}).get(proto)
                    require(expected is not None and expected.get("status") == "PASS",
                            f"No matching PASS in P3 verification report for {proto}")
                    require(expected["query_images"] == len(queries) and
                            expected["gallery_images"] == len(gallery),
                            f"P3 verification counts disagree: {proto}")
                    info = protocol_report.get("protocols", {}).get(proto)
                    require(info and info.get("query_images") == len(queries) and
                            info.get("gallery_images") == len(gallery),
                            f"P3 retrieval protocol report disagree: {proto}")
                    protocol_stats[proto] = {
                        "dataset": PROTOCOL_DATASETS[proto],
                        "query_images": len(queries), "gallery_images": len(gallery),
                        "query_identities": len(set(r["identity_key"] for r in queries)),
                        "gallery_identities": len(set(r["identity_key"] for r in gallery)),
                        "rule": info.get("rule"),
                        "source": "provided_p3_retrieval_protocol_report",
                        "status": "internally_verified_not_official_test",
                    }
    
                # Every P3 validation row must be represented in at least one retrieval protocol.
                for dataset, count in source_counts.items():
                    unrepresented = db.execute(
                        "SELECT COUNT(*) FROM samples s WHERE s.dataset=? AND s.role='validation' "
                        "AND NOT EXISTS (SELECT 1 FROM retrieval r WHERE r.uid=s.uid)",
                        (dataset,)).fetchone()[0]
                    require(unrepresented == 0,
                            f"{dataset}: {unrepresented} P3 validation images missing from retrieval CSVs")
                    unexpected = db.execute(
                        "SELECT COUNT(*) FROM samples s WHERE s.dataset=? AND s.role='train' "
                        "AND EXISTS (SELECT 1 FROM retrieval r WHERE r.uid=s.uid)",
                        (dataset,)).fetchone()[0]
                    require(unexpected == 0, f"{dataset}: train/validation image overlap")
                    if count.get("validation", 0) == 0:
                        require(not any(v["dataset"] == dataset for v in protocol_stats.values()),
                                f"{dataset}: retrieval protocol despite no validation split")
    
                # C) Write agreed output shapes (never give member-specific global class IDs).
                with (stage / "train_manifest.csv").open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=TRAIN_COLS, lineterminator="\n")
                    writer.writeheader()
                    for (raw,) in db.execute("SELECT data FROM samples WHERE role='train' ORDER BY dataset,uid"):
                        row = json.loads(raw)
                        out = {k: row[k] for k in TRAIN_COLS if k != "class_id"}
                        out["class_id"] = identity_to_class[row["identity_key"]]
                        writer.writerow(out)
                with (stage / "validation_manifest.csv").open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=VAL_COLS, lineterminator="\n")
                    writer.writeheader()
                    for proto, uid, role, raw in db.execute(
                        "SELECT r.protocol,r.uid,r.role,s.data FROM retrieval r "
                        "JOIN samples s ON r.uid=s.uid ORDER BY r.protocol,r.role,r.uid"):
                        row = json.loads(raw)
                        out = {k: row[k] for k in VAL_COLS if k not in
                               ("validation_protocol_id", "retrieval_role")}
                        out["validation_protocol_id"] = proto
                        out["retrieval_role"] = role
                        writer.writerow(out)
                write_json(stage / "identity_class_mapping.json", {
                    "schema_version": "v1", "num_classes": len(identity_to_class),
                    "identity_to_class": identity_to_class,
                })
                write_json(stage / "source_registry.json", {
                    "schema_version": "v1", "sources": [
                        {"dataset": name,
                         "dataset_root_id": sorted(source_roots[name])[0],
                         "root_env_var": "DGREID_ROOT_" + re.sub(r"[^A-Z0-9]+", "_", sorted(source_roots[name])[0].upper()),
                         "relative_base_path": "",
                         "train_rows": source_counts[name].get("train", 0),
                         "validation_unique_rows": source_counts[name].get("validation", 0)}
                        for name in sorted(source_counts)
                    ]
                })
                # Recorded evidence can be upgraded only after explicit P3/P4 scientific approval.
                approval = None
                if args.approval_json:
                    with args.approval_json.open("r", encoding="utf-8") as f:
                        approval = json.load(f)
                    require(approval.get("p3_approved") is True and approval.get("p4_cleared") is True,
                            "Approval file must explicitly certify P3 and P4")
                    require(set(approval.get("approved_datasets", [])) == set(source_counts),
                            "Approval dataset coverage mismatch")
                    require(approval.get("approved_by") and approval.get("evidence_files"),
                            "Approval requires reviewer and evidence file hashes")
                    for item in approval["evidence_files"]:
                        file = Path(item["path"])
                        require(file.is_file() and file_sha256(file) == item["sha256"],
                                f"Approval evidence missing/modified: {file}")
                    require(not approval.get("excluded_sample_uids"),
                            "P4 exclusions must be applied through revised P3 inputs before P5. "
                            "Do not silently drop images here.")
                status = "P3_P4_APPROVED_METADATA_ONLY" if approval else "PROVISIONAL_P3_P4_PENDING"
                protocol_json = {
                    "schema_version": "v1",
                    "protocol_id": "dg_reid_multi_source_p5_v1",
                    "status": status,
                    "p3_p4_externally_approved": bool(approval),
                    "ready_for_finetuning": False,  # image access/checkpoint readiness must be checked in P7
                    "scope": "Only the selected datasets; NOT all nine unless --datasets all was supplied and validated",
                    "selected_datasets": sorted(source_counts),
                    "num_selected_sources": len(source_counts),
                    "expected_full_program_sources": 9,
                    "data_images_and_paths_verified": False,
                    "unified_master_sha256": None,
                    "source_versions": inputs,
                    "split_policy": {
                        "method": "Reused P3 per-image roles verbatim; never inferred from retrieval complement",
                        "p3_source_role_counts": source_counts,
                        "identity_disjointness_checked": True,
                        "no_train_retrieval_overlap_checked": True,
                        "seed_reported_by_p3": protocol_report.get("seed"),
                    },
                    "training_sources": sorted(name for name, stat in source_counts.items() if stat.get("train", 0)),
                    "validation_protocols": protocol_stats,
                    "exclusion_policy": {
                        "p4_status": "EXTERNAL_APPROVAL_RECORDED" if approval else "NOT_CLEARED",
                        "p4_approval_evidence": approval,
                        "note": "No new exclusions invented or applied in P5; P4 must be completed separately",
                    },
                    "target_datasets": ["Market1501", "MARS"],
                    "caveats": [
                        "Metadata and P3 retrieval checks cannot prove source-to-source/target acquisition independence.",
                        "Actual image bytes, runtime root paths, and pretrained checkpoint provenance were not checked.",
                        "No final nine-source claim if only a subset of sources was supplied.",
                    ],
                }
                write_json(stage / "dataset_protocol.json", protocol_json)
                with (stage / "checksums.sha256").open("w", encoding="utf-8", newline="\n") as f:
                    for filename in OUTPUT_NAMES[:-1]:
                        f.write(f"{file_sha256(stage / filename)}  {filename}\n")
            finally:
                db.close()
            (stage / "_p5_build.sqlite").unlink()
            # Temp staging folder is on same volume, no overwrite / partial final outputs.
            stage.rename(output)
            print("\nP5 BUILD: PASS (metadata-level)")
            print("Status:", status)
            print("Selected datasets:", ", ".join(sorted(source_counts)))
            print("Training images:", sum(x.get("train", 0) for x in source_counts.values()))
            print("Training identities:", len(identity_to_class))
            print("Unique validation images:", sum(x.get("validation", 0) for x in source_counts.values()))
            print("Retrieval manifest records:", sum(v["query_images"] + v["gallery_images"] for v in protocol_stats.values()))
            print("Output:", output)
    finally:
        retrieval.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Create the six P5 output files with strict P3 checks")
    parser.add_argument("--unified-dir", type=Path, required=True,
                        help="Directory containing frozen per-source *_unified_v1.csv files")
    parser.add_argument("--p3-dir", type=Path, required=True,
                        help="Folder containing COMPLETE per-image *_p3_provisional.csv role assignments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--retrieval-dir", type=Path, help="Extracted folder containing retrieval CSVs + P3 JSON reports")
    group.add_argument("--retrieval-zip", type=Path, help="ZIP containing retrieval CSVs + P3 JSON reports")
    parser.add_argument("--output-dir", type=Path, default=Path("final_dataset/v1"))
    parser.add_argument("--datasets", nargs="+", default=["cuhk03", "deepchange", "occluded_posetrack"],
                        help="Selected keys, or `all` for all nine (requires every corresponding P3 input)")
    parser.add_argument("--approval-json", type=Path, help="Optional externally APPROVED P3/P4 evidence manifest")
    return parser.parse_args()

if __name__ == "__main__":
    try:
        build(parse_args())
    except (DataError, ValueError, OSError, sqlite3.Error, zipfile.BadZipFile) as e:
        print(f"P5 BUILD FAILED: {e}", file=sys.stderr)
        sys.exit(1)
