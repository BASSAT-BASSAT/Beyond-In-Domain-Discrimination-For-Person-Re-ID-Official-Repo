from pathlib import Path
import numpy as np
from scipy.io import loadmat


BASE = Path.home() / "DG_ReID_Audit"

ROOT = (
    BASE
    / "datasets/PRW/extracted"
)


def public_keys(mat):
    return [
        k for k in mat.keys()
        if not k.startswith("__")
    ]


def describe_array(name, value):

    arr = np.asarray(value)

    print(f"\n{name}")
    print("-" * 70)

    print("type :", type(value))
    print("dtype:", arr.dtype)
    print("shape:", arr.shape)

    flat = arr.flatten()

    print(
        "first values:",
        flat[:20]
    )


print("=" * 90)
print("PRW METADATA / SPLIT INSPECTION")
print("=" * 90)


# ============================================================
# ROOT MAT FILES
# ============================================================

for filename in [
    "ID_train.mat",
    "ID_test.mat",
    "frame_train.mat",
    "frame_test.mat",
]:

    path = ROOT / filename

    mat = loadmat(
        path,
        squeeze_me=True,
        chars_as_strings=True,
    )

    print("\n" + "=" * 90)
    print(filename)
    print("=" * 90)

    print(
        "Keys:",
        public_keys(mat)
    )

    for key in public_keys(mat):
        describe_array(
            key,
            mat[key]
        )


# ============================================================
# SAMPLE ANNOTATIONS
# ============================================================

annotation_dir = (
    ROOT
    / "annotations"
)

annotation_files = sorted(
    annotation_dir.glob(
        "*.mat"
    )
)


print("\n" + "=" * 90)
print("ANNOTATION FILES")
print("=" * 90)

print(
    "Count:",
    len(annotation_files)
)


for path in annotation_files[:5]:

    mat = loadmat(
        path,
        squeeze_me=True,
        chars_as_strings=True,
    )

    print(
        f"\n{path.name}"
    )

    print(
        "Keys:",
        public_keys(mat)
    )

    for key in public_keys(mat):

        arr = np.asarray(
            mat[key]
        )

        print(
            f"{key}: "
            f"shape={arr.shape}, "
            f"dtype={arr.dtype}"
        )

        print(
            arr
        )


# ============================================================
# QUERY INFO
# ============================================================

query_info = (
    ROOT
    / "query_info.txt"
)


print("\n" + "=" * 90)
print("QUERY INFO")
print("=" * 90)


lines = query_info.read_text(
    encoding="utf-8",
    errors="replace",
).splitlines()


print(
    "Rows:",
    len(lines)
)

print(
    "\nFirst 20 rows:"
)

for line in lines[:20]:
    print(line)


# ============================================================
# QUERY BOX FILENAMES
# ============================================================

query_dir = (
    ROOT
    / "query_box"
)

queries = sorted(
    query_dir.glob(
        "*.jpg"
    )
)


print("\n" + "=" * 90)
print("QUERY BOXES")
print("=" * 90)

print(
    "Count:",
    len(queries)
)

print(
    "First 20:"
)

for path in queries[:20]:
    print(
        path.name
    )


print("\nDONE")
