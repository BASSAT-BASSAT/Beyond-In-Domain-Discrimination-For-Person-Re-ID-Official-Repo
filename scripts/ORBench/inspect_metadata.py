from pathlib import Path
import json

ROOT = (
    Path.home()
    / "DG_ReID_Audit/datasets/ORBench/extracted/ORBench"
)

files = [
    ROOT / "train_annos.json",
    ROOT / "test_gallery_and_queries.json",
]

for path in files:
    print("\n" + "=" * 100)
    print(path.name)
    print("=" * 100)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Top-level type:", type(data).__name__)

    if isinstance(data, dict):
        print("Top-level keys:")
        for key in data.keys():
            print("  ", repr(key))

        print("\nKey summaries:")

        for key, value in data.items():
            print(
                f"{key!r:<35} "
                f"type={type(value).__name__:<10}",
                end=""
            )

            if hasattr(value, "__len__"):
                print(f" len={len(value):,}")
            else:
                print()

            if isinstance(value, list) and value:
                print("   first item type:", type(value[0]).__name__)
                print("   first item:", repr(value[0])[:1000])

            elif isinstance(value, dict) and value:
                first_key = next(iter(value))
                print("   first key:", repr(first_key))
                print(
                    "   first value:",
                    repr(value[first_key])[:1000]
                )

    elif isinstance(data, list):
        print("Length:", len(data))

        if data:
            print("First item type:", type(data[0]).__name__)
            print("First item:", repr(data[0])[:1500])

    print()
