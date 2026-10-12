from pathlib import Path
from collections import Counter, defaultdict
import json
import re

ROOT = (
    Path.home()
    / "DG_ReID_Audit/datasets/ORBench/extracted/ORBench"
)

TRAIN_JSON = ROOT / "train_annos.json"
TEST_JSON = ROOT / "test_gallery_and_queries.json"


# ============================================================
# HELPERS
# ============================================================

def parse_path(path_str):
    """
    Parse paths such as:

    vis/0001/0001_llcm_0001_c03_s184251_f31995_vis.jpg
    nir/0005/0005_llcm_0006_c06_s170450_f2220_nir.jpg
    cp/0006/0006_llcm_0008_back_2_colorpencil.jpg
    sk/0760/0760_sysu_0288_front_4_sketch.jpg
    """

    p = Path(path_str)

    modality = p.parts[0]
    folder_id = int(p.parts[1])
    name = p.name

    tokens = name.split("_")

    global_id = int(tokens[0])
    source = tokens[1]
    source_pid = int(tokens[2])

    camera = None
    sequence = None
    frame = None
    view = None
    variant_index = None

    for t in tokens[3:]:

        stem = t.rsplit(".", 1)[0]

        if re.fullmatch(r"c\d+", stem):
            camera = int(stem[1:])

        elif re.fullmatch(r"s\d+", stem):
            sequence = int(stem[1:])

        elif re.fullmatch(r"f\d+", stem):
            frame = int(stem[1:])

        elif stem in {"front", "back", "side"}:
            view = stem

    # CP/SK usually have ... front_0_...
    if modality in {"cp", "sk"}:
        for i, t in enumerate(tokens):
            if t in {"front", "back", "side"}:
                if i + 1 < len(tokens):
                    try:
                        variant_index = int(tokens[i + 1])
                    except ValueError:
                        pass

    return {
        "modality": modality,
        "folder_id": folder_id,
        "global_id": global_id,
        "source": source,
        "source_pid": source_pid,
        "camera": camera,
        "sequence": sequence,
        "frame": frame,
        "view": view,
        "variant_index": variant_index,
    }


def summarize_paths(name, records):

    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)

    parsed = []
    missing = 0
    parse_errors = []

    for label, path_str in records:

        full = ROOT / path_str

        if not full.is_file():
            missing += 1

        try:
            x = parse_path(path_str)
            x["label"] = int(label)
            x["path"] = path_str
            parsed.append(x)

        except Exception as e:
            parse_errors.append(
                (path_str, repr(e))
            )

    print("Records:", len(records))
    print("Existing files:", len(records) - missing)
    print("Missing files:", missing)
    print("Parse errors:", len(parse_errors))

    if parse_errors:
        print("\nFirst parse errors:")
        for item in parse_errors[:20]:
            print(item)

    if not parsed:
        return []

    labels = {x["label"] for x in parsed}
    gids = {x["global_id"] for x in parsed}

    print("\nUnique JSON labels:", len(labels))
    print(
        "Label range:",
        min(labels),
        "->",
        max(labels)
    )

    print(
        "Unique filename global IDs:",
        len(gids)
    )

    print(
        "Global ID range:",
        min(gids),
        "->",
        max(gids)
    )

    label_gid_mismatch = sum(
        x["label"] != x["global_id"]
        for x in parsed
    )

    folder_gid_mismatch = sum(
        x["folder_id"] != x["global_id"]
        for x in parsed
    )

    print(
        "JSON label != filename global ID:",
        label_gid_mismatch
    )

    print(
        "Folder ID != filename global ID:",
        folder_gid_mismatch
    )

    print("\nSource dataset counts (images):")

    print(
        Counter(
            x["source"]
            for x in parsed
        )
    )

    print("\nSource dataset unique global IDs:")

    by_source_ids = defaultdict(set)

    for x in parsed:
        by_source_ids[
            x["source"]
        ].add(
            x["global_id"]
        )

    for source in sorted(by_source_ids):
        ids = by_source_ids[source]

        print(
            f"{source}: "
            f"{len(ids)} IDs "
            f"[{min(ids)}..{max(ids)}]"
        )

    print("\nOriginal source identities:")

    original_keys = {
        (
            x["source"],
            x["source_pid"]
        )
        for x in parsed
    }

    print(
        "Unique (source, source_pid):",
        len(original_keys)
    )

    if any(
        x["camera"] is not None
        for x in parsed
    ):

        cams = Counter(
            x["camera"]
            for x in parsed
            if x["camera"] is not None
        )

        print("\nCamera counts:")
        print(dict(sorted(cams.items())))

    return parsed


# ============================================================
# LOAD JSON
# ============================================================

with open(
    TRAIN_JSON,
    "r",
    encoding="utf-8"
) as f:
    train = json.load(f)

with open(
    TEST_JSON,
    "r",
    encoding="utf-8"
) as f:
    test = json.load(f)


print("=" * 100)
print("ORBENCH PROTOCOL + ORIGIN AUDIT")
print("=" * 100)


# ============================================================
# TRAIN JSON
# ============================================================

print("\nTRAIN JSON")

print("Records:", len(train))

print(
    "Split values:",
    Counter(
        x.get("split")
        for x in train
    )
)

print(
    "Path modalities:",
    Counter(
        Path(x["file_path"]).parts[0]
        for x in train
    )
)

train_records = [
    (
        x["id"],
        x["file_path"]
    )
    for x in train
]


train_parsed = summarize_paths(
    "TRAIN PATH AUDIT",
    train_records
)


# ============================================================
# TEST RGB GALLERY
# ============================================================

gallery_records = [
    (
        item[0],
        item[1]
    )
    for item in test["RGB_GALLERY"]
]


gallery_parsed = summarize_paths(
    "RGB GALLERY PATH AUDIT",
    gallery_records
)


# ============================================================
# SINGLE-MODAL TEST QUERIES
# ============================================================

single_keys = [
    "NIR",
    "CP",
    "SK",
]


single_results = {}


for key in single_keys:

    records = [
        (
            item[0],
            item[1]
        )
        for item in test[key]
    ]

    single_results[key] = summarize_paths(
        f"{key} QUERY PATH AUDIT",
        records
    )


# ============================================================
# TRAIN / TEST GLOBAL ID LEAKAGE
# ============================================================

train_ids = {
    x["global_id"]
    for x in train_parsed
}

gallery_ids = {
    x["global_id"]
    for x in gallery_parsed
}


print("\n" + "=" * 100)
print("TRAIN / TEST IDENTITY SEPARATION")
print("=" * 100)


print(
    "Train IDs:",
    len(train_ids),
    (
        min(train_ids),
        max(train_ids)
    )
)

print(
    "Test RGB gallery IDs:",
    len(gallery_ids),
    (
        min(gallery_ids),
        max(gallery_ids)
    )
)


overlap = (
    train_ids
    &
    gallery_ids
)


print(
    "Global-ID overlap:",
    len(overlap)
)

if overlap:
    print(
        "Overlapping IDs:",
        sorted(overlap)[:50]
    )


# ============================================================
# ORIGINAL IDENTITY LEAKAGE
#
# Important because global ORBench IDs are remapped.
# Compare original dataset + original PID.
# ============================================================

train_original = {
    (
        x["source"],
        x["source_pid"]
    )
    for x in train_parsed
}


test_original = {
    (
        x["source"],
        x["source_pid"]
    )
    for x in gallery_parsed
}


original_overlap = (
    train_original
    &
    test_original
)


print("\n" + "=" * 100)
print("ORIGINAL SOURCE-IDENTITY LEAKAGE")
print("=" * 100)


print(
    "Train original identities:",
    len(train_original)
)

print(
    "Test original identities:",
    len(test_original)
)

print(
    "Overlap in (source_dataset, original_pid):",
    len(original_overlap)
)


if original_overlap:

    print("\nFirst overlaps:")

    for item in sorted(
        original_overlap
    )[:50]:

        print(item)


# ============================================================
# QUERY ID CONSISTENCY
# ============================================================

print("\n" + "=" * 100)
print("TEST MODALITY ID CONSISTENCY")
print("=" * 100)


for key, parsed in single_results.items():

    ids = {
        x["global_id"]
        for x in parsed
    }

    missing_from_gallery = (
        ids
        - gallery_ids
    )

    missing_from_query = (
        gallery_ids
        - ids
    )

    print(
        f"{key}:"
    )

    print(
        "  unique IDs:",
        len(ids)
    )

    print(
        "  IDs not in RGB gallery:",
        len(
            missing_from_gallery
        )
    )

    print(
        "  gallery IDs not in modality:",
        len(
            missing_from_query
        )
    )


# ============================================================
# FILESYSTEM ID FOLDERS
# ============================================================

print("\n" + "=" * 100)
print("FILESYSTEM MODALITY SUMMARY")
print("=" * 100)


for modality in [
    "vis",
    "nir",
    "cp",
    "sk",
]:

    root = ROOT / modality

    folders = sorted(
        p
        for p in root.iterdir()
        if p.is_dir()
    )

    images = list(
        root.glob("*/*.jpg")
    )

    print(
        f"{modality.upper():<4} "
        f"folders={len(folders):4d} "
        f"images={len(images):6d}"
    )


print("\nDONE")
