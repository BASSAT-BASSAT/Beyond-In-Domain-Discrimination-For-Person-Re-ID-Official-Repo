from pathlib import Path
from collections import Counter
import re

ROOT = Path.home() / "DG_ReID_Audit/datasets/LaST/extracted/last"

normal = re.compile(
    r"^(?P<pid>\d{6})_"
    r"(?P<image_id>\d{3})_"
    r"(?P<video_id>\d{5})_"
    r"(?P<frame_id>\d+)_"
    r"(?P<bbox_id>\d+)_"
    r"(?P<clothes_id>\d{3})\.jpg$"
)

sources = [
    ("train", ROOT / "train"),
    ("val_query", ROOT / "val/query"),
    ("val_gallery", ROOT / "val/gallery"),
    ("test_query", ROOT / "test/query"),
    ("test_gallery", ROOT / "test/gallery"),
]

invalid = []

for split, base in sources:
    for p in sorted(base.rglob("*.jpg")):
        if not normal.match(p.name):
            parts = p.stem.split("_")

            invalid.append({
                "split": split,
                "filename": p.name,
                "parts_count": len(parts),
                "parts": parts,
                "parent_folder": p.parent.name,
            })

print("=" * 80)
print("INVALID FILENAME DIAGNOSTIC")
print("=" * 80)

print("\nTotal invalid:", len(invalid))

print("\nInvalid by split:")
print(Counter(x["split"] for x in invalid))

print("\nParts-count distribution:")
print(Counter(x["parts_count"] for x in invalid))

print("\nAll invalid filenames:")
for x in invalid:
    print()
    print("split        :", x["split"])
    print("folder       :", x["parent_folder"])
    print("filename     :", x["filename"])
    print("parts_count  :", x["parts_count"])
    print("parts        :", x["parts"])
