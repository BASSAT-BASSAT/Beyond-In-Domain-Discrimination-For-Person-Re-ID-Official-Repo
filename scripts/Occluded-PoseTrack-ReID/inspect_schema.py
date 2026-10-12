from pathlib import Path
import json
import re


ROOT = (
    Path.home()
    / "DG_ReID_Audit/datasets/Occluded-PoseTrack-ReID"
    / "extracted/occluded_posetrack_reid"
)

IMAGES = ROOT / "images"


# ============================================================
# Helpers
# ============================================================

def short_summary(value):
    """
    Describe a JSON value without dumping huge arrays.
    """

    if isinstance(value, dict):
        return {
            "type": "dict",
            "length": len(value),
            "keys": list(value.keys())[:30],
        }

    if isinstance(value, list):

        preview = value[:8]

        return {
            "type": "list",
            "length": len(value),
            "preview": preview,
        }

    return value


def summarize_record(record):
    """
    Show all fields in one representative record,
    but summarize large lists/dicts.
    """

    if not isinstance(record, dict):
        return short_summary(record)

    result = {}

    for key, value in record.items():

        if isinstance(value, list):
            result[key] = {
                "type": "list",
                "length": len(value),
                "preview": value[:8],
            }

        elif isinstance(value, dict):
            result[key] = {
                "type": "dict",
                "length": len(value),
                "keys": list(value.keys())[:20],
            }

        else:
            result[key] = value

    return result


# ============================================================
# 1. Physical image structure
# ============================================================

pattern = re.compile(
    r"^(?P<field1>\d+)_(?P<field2>\d+)_(?P<field3>\d+)\.jpg$",
    re.IGNORECASE,
)


print("=" * 90)
print("PHYSICAL IMAGE STRUCTURE")
print("=" * 90)


for split in ["train", "val"]:

    split_dir = IMAGES / split

    folders = sorted(
        p
        for p in split_dir.iterdir()
        if p.is_dir()
    )

    files = sorted(
        split_dir.glob("*/*.jpg")
    )

    parsed = []
    invalid = []

    folder_field2_matches = 0

    for path in files:

        m = pattern.match(path.name)

        if not m:
            invalid.append(path)
            continue

        field1 = m.group("field1")
        field2 = m.group("field2")
        field3 = m.group("field3")

        if path.parent.name == field2:
            folder_field2_matches += 1

        parsed.append(
            {
                "folder": path.parent.name,
                "field1": field1,
                "field2": field2,
                "field3": field3,
                "filename": path.name,
            }
        )

    print("\n" + "-" * 90)
    print(split.upper())
    print("-" * 90)

    print("Folders                 :", len(folders))
    print("JPG images              :", len(files))
    print("Parsed filenames        :", len(parsed))
    print("Invalid filenames       :", len(invalid))

    if parsed:

        field1_values = {
            x["field1"]
            for x in parsed
        }

        field2_values = {
            x["field2"]
            for x in parsed
        }

        field3_values = {
            x["field3"]
            for x in parsed
        }

        print("Unique field1 values    :", len(field1_values))
        print("Unique field2 values    :", len(field2_values))
        print("Unique field3 values    :", len(field3_values))

        print(
            "Folder == field2        :",
            f"{folder_field2_matches} / {len(parsed)}",
            f"({folder_field2_matches / len(parsed) * 100:.2f}%)",
        )

        print("\nExample filenames:")

        for x in parsed[:10]:
            print(
                f"  folder={x['folder']} | "
                f"f1={x['field1']} | "
                f"f2={x['field2']} | "
                f"f3={x['field3']} | "
                f"{x['filename']}"
            )

    if invalid:

        print("\nInvalid examples:")

        for path in invalid[:10]:
            print(" ", path)


# ============================================================
# 2. JSON structure
# ============================================================

JSON_FILES = [
    IMAGES / "anns/train_reid_crops_anns.json",
    IMAGES / "anns/val_reid_crops_anns.json",
    ROOT / "train_dataset_sampling.json",
    ROOT / "val_dataset_sampling.json",
]


print("\n\n" + "=" * 90)
print("JSON SCHEMA INSPECTION")
print("=" * 90)


for json_path in JSON_FILES:

    print("\n" + "=" * 90)
    print(json_path.name)
    print("=" * 90)

    size_mb = json_path.stat().st_size / (1024 ** 2)

    print(f"File size : {size_mb:.2f} MiB")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Root type :", type(data).__name__)

    # --------------------------------------------------------
    # Root dictionary
    # --------------------------------------------------------

    if isinstance(data, dict):

        print("Root keys :", list(data.keys()))
        print("Root length:", len(data))

        for key, value in data.items():

            print(
                f"\n[{key}] "
                f"type={type(value).__name__}",
                end=""
            )

            if hasattr(value, "__len__"):
                print(f" length={len(value)}")
            else:
                print()

            # list → inspect first record
            if isinstance(value, list) and value:

                print("First item:")

                print(
                    json.dumps(
                        summarize_record(value[0]),
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                )

            # dict → inspect first key/value
            elif isinstance(value, dict) and value:

                first_key = next(iter(value))
                first_value = value[first_key]

                print("First dictionary key:")
                print(" ", repr(first_key))

                print("First dictionary value:")

                print(
                    json.dumps(
                        summarize_record(first_value),
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                )

            else:

                print(
                    "Value:",
                    repr(value)[:1000]
                )

    # --------------------------------------------------------
    # Root list
    # --------------------------------------------------------

    elif isinstance(data, list):

        print("Root length:", len(data))

        if data:

            print("\nFirst item:")

            print(
                json.dumps(
                    summarize_record(data[0]),
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )

            if len(data) > 1:

                print("\nSecond item:")

                print(
                    json.dumps(
                        summarize_record(data[1]),
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                )

    else:

        print("Root value:", repr(data)[:2000])


print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
