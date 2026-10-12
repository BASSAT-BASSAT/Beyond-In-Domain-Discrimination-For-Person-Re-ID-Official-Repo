from pathlib import Path
from collections import Counter, defaultdict
import json


ROOT = (
    Path.home()
    / "DG_ReID_Audit/datasets/Occluded-PoseTrack-ReID"
    / "extracted/occluded_posetrack_reid"
)

IMAGES = ROOT / "images"


def is_real_path(value):
    return (
        isinstance(value, str)
        and value.strip() != ""
    )


def analyze(name):

    ann_path = (
        IMAGES
        / "anns"
        / f"{name}_reid_crops_anns.json"
    )

    sampling_path = (
        ROOT
        / f"{name}_dataset_sampling.json"
    )

    physical_dir = IMAGES / name


    # ========================================================
    # Load JSONs
    # ========================================================

    with open(
        ann_path,
        "r",
        encoding="utf-8"
    ) as f:
        anns = json.load(f)

    with open(
        sampling_path,
        "r",
        encoding="utf-8"
    ) as f:
        sampling = json.load(f)


    print("\n" + "=" * 90)
    print(name.upper())
    print("=" * 90)


    # ========================================================
    # 1. Check JSON key consistency
    # ========================================================

    ann_id_keys = set(
        anns["id"].keys()
    )

    crop_path_keys = set(
        anns["reid_crop_path"].keys()
    )

    width_keys = set(
        anns["reid_crop_width"].keys()
    )

    height_keys = set(
        anns["reid_crop_height"].keys()
    )

    kps_keys = set(
        anns["negative_kps"].keys()
    )

    sampling_id_keys = set(
        sampling["id"].keys()
    )

    sampling_split_keys = set(
        sampling["split"].keys()
    )


    print("\nJSON RECORD COUNTS")
    print("-" * 90)

    print(
        "Annotation IDs          :",
        len(ann_id_keys)
    )

    print(
        "Sampling IDs            :",
        len(sampling_id_keys)
    )


    print("\nKEY CONSISTENCY")
    print("-" * 90)

    print(
        "ann id == crop path     :",
        ann_id_keys == crop_path_keys
    )

    print(
        "ann id == width         :",
        ann_id_keys == width_keys
    )

    print(
        "ann id == height        :",
        ann_id_keys == height_keys
    )

    print(
        "ann id == negative_kps  :",
        ann_id_keys == kps_keys
    )

    print(
        "ann ids == sampling ids :",
        ann_id_keys == sampling_id_keys
    )

    print(
        "sampling id == split    :",
        sampling_id_keys == sampling_split_keys
    )


    # ========================================================
    # 2. Sampling split distribution
    # ========================================================

    split_counts = Counter(
        sampling["split"].values()
    )

    print("\nSAMPLING SPLIT DISTRIBUTION")
    print("-" * 90)

    for label, count in sorted(
        split_counts.items(),
        key=lambda x: str(x[0])
    ):

        pct = (
            count
            / len(sampling["split"])
            * 100
        )

        print(
            f"{repr(label):<20}"
            f"{count:>8} "
            f"({pct:6.2f}%)"
        )


    # ========================================================
    # 3. Which annotations actually have a ReID crop?
    # ========================================================

    crop_records = []

    non_null_by_split = Counter()

    for record_id in ann_id_keys:

        crop_path = (
            anns["reid_crop_path"]
            .get(record_id)
        )

        split_label = (
            sampling["split"]
            .get(record_id)
        )

        if is_real_path(crop_path):

            crop_records.append(
                {
                    "id": record_id,
                    "split": split_label,
                    "path": crop_path,
                    "width":
                        anns["reid_crop_width"]
                        .get(record_id),

                    "height":
                        anns["reid_crop_height"]
                        .get(record_id),
                }
            )

            non_null_by_split[
                split_label
            ] += 1


    print("\nREID CROPS IN ANNOTATIONS")
    print("-" * 90)

    print(
        "Records with crop path  :",
        len(crop_records)
    )

    print(
        "Records without crop    :",
        len(ann_id_keys)
        - len(crop_records)
    )


    print("\nCrop paths by sampling split:")

    for label, count in sorted(
        non_null_by_split.items(),
        key=lambda x: str(x[0])
    ):

        print(
            f"{repr(label):<20}"
            f"{count:>8}"
        )


    # ========================================================
    # 4. Physical JPG files
    # ========================================================

    physical_files = sorted(
        physical_dir.glob("*/*.jpg")
    )

    physical_names = Counter(
        p.name
        for p in physical_files
    )


    print("\nPHYSICAL JPG FILES")
    print("-" * 90)

    print(
        "Physical images         :",
        len(physical_files)
    )

    print(
        "Unique basenames        :",
        len(physical_names)
    )

    duplicate_physical_names = {
        name: count
        for name, count
        in physical_names.items()
        if count > 1
    }

    print(
        "Duplicate basenames     :",
        len(duplicate_physical_names)
    )


    # ========================================================
    # 5. JSON crop path → physical JPG matching
    # ========================================================

    crop_basenames = Counter(
        Path(
            record["path"]
            .replace("\\", "/")
        ).name

        for record in crop_records
    )


    matched_unique = 0
    ambiguous = 0
    missing = 0

    missing_examples = []

    for basename, json_count in (
        crop_basenames.items()
    ):

        physical_count = (
            physical_names.get(
                basename,
                0
            )
        )

        if physical_count == 1:
            matched_unique += 1

        elif physical_count > 1:
            ambiguous += 1

        else:
            missing += 1

            if len(missing_examples) < 10:
                missing_examples.append(
                    basename
                )


    print("\nJSON ↔ PHYSICAL IMAGE MATCH")
    print("-" * 90)

    print(
        "Unique JSON crop names  :",
        len(crop_basenames)
    )

    print(
        "Matched uniquely        :",
        matched_unique
    )

    print(
        "Ambiguous basename      :",
        ambiguous
    )

    print(
        "JSON names not physical :",
        missing
    )


    referenced_physical = {
        basename
        for basename
        in crop_basenames
        if basename in physical_names
    }

    physical_unreferenced = (
        set(physical_names)
        - referenced_physical
    )

    print(
        "Physical not referenced :",
        len(physical_unreferenced)
    )


    if missing_examples:

        print("\nMissing JSON basename examples:")

        for x in missing_examples:
            print(" ", x)


    # ========================================================
    # 6. Show representative selected records
    # ========================================================

    print("\nSELECTED RECORD EXAMPLES")
    print("-" * 90)

    examples_by_split = defaultdict(list)

    for record in crop_records:

        label = record["split"]

        if len(
            examples_by_split[label]
        ) < 5:

            examples_by_split[
                label
            ].append(record)


    for label in sorted(
        examples_by_split,
        key=lambda x: str(x)
    ):

        print(
            f"\nSplit = {repr(label)}"
        )

        for record in (
            examples_by_split[label]
        ):

            print(
                "  "
                f"id={record['id']} | "
                f"{record['width']}x"
                f"{record['height']} | "
                f"path={record['path']}"
            )


    # ========================================================
    # 7. Check crop dimensions null/non-null consistency
    # ========================================================

    crop_with_null_width = sum(
        record["width"] is None
        for record in crop_records
    )

    crop_with_null_height = sum(
        record["height"] is None
        for record in crop_records
    )

    print("\nCROP METADATA CONSISTENCY")
    print("-" * 90)

    print(
        "Crop path + null width  :",
        crop_with_null_width
    )

    print(
        "Crop path + null height :",
        crop_with_null_height
    )


for split_name in [
    "train",
    "val",
]:

    analyze(split_name)


print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
