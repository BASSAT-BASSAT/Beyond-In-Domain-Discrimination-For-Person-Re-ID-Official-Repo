from pathlib import Path

import deeplake
import numpy as np


ROOT = (
    Path.home()
    / "DG_ReID_Audit"
    / "datasets"
    / "MARS"
)


SPLITS = [
    {
        "name": "TRAIN",
        "dataset": "hub://activeloop/mars-train",
        "official_names": ROOT / "info/train_name.txt",
        "expected": 509_914,
    },
    {
        "name": "TEST",
        "dataset": "hub://activeloop/mars-test",
        "official_names": ROOT / "info/test_name.txt",
        "expected": 681_089,
    },
]


def load_names(path):
    return [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def get_class_names(tensor):

    info = tensor.info

    try:
        return list(
            info["class_names"]
        )
    except Exception:
        pass

    try:
        return list(
            info.class_names
        )
    except Exception:
        pass

    raise RuntimeError(
        "Could not read class_names "
        f"for tensor {tensor}"
    )


def get_codes(ds, key):

    values = (
        ds[key][:]
        .numpy()
        .reshape(-1)
    )

    return values.astype(
        np.int64
    )


def audit_split(cfg):

    print("\n" + "=" * 100)
    print(
        f"MARS {cfg['name']} "
        f"DEEPLAKE ↔ OFFICIAL NAME CHECK"
    )
    print("=" * 100)

    official = load_names(
        cfg["official_names"]
    )

    print(
        "Official names:",
        f"{len(official):,}"
    )

    ds = deeplake.load(
        cfg["dataset"],
        read_only=True,
    )

    print(
        "Deep Lake samples:",
        f"{len(ds):,}"
    )

    print(
        "Expected:",
        f"{cfg['expected']:,}"
    )


    if len(ds) != cfg["expected"]:
        raise RuntimeError(
            "Unexpected Deep Lake size."
        )

    if len(official) != cfg["expected"]:
        raise RuntimeError(
            "Unexpected official name count."
        )


    required = [
        "pedestrian_ids",
        "camera_nos",
        "track_nos",
        "frame_nos",
    ]


    for key in required:

        if key not in ds.tensors:
            raise RuntimeError(
                f"Missing tensor: {key}"
            )


    print(
        "\nReading encoded metadata..."
    )


    pid_codes = get_codes(
        ds,
        "pedestrian_ids"
    )

    camera_codes = get_codes(
        ds,
        "camera_nos"
    )

    track_codes = get_codes(
        ds,
        "track_nos"
    )

    frame_codes = get_codes(
        ds,
        "frame_nos"
    )


    pid_names = get_class_names(
        ds["pedestrian_ids"]
    )

    camera_names = get_class_names(
        ds["camera_nos"]
    )

    track_names = get_class_names(
        ds["track_nos"]
    )

    frame_names = get_class_names(
        ds["frame_nos"]
    )


    reconstructed = []


    print(
        "Reconstructing filenames..."
    )


    for i in range(len(ds)):

        filename = (
            pid_names[
                pid_codes[i]
            ]
            +
            camera_names[
                camera_codes[i]
            ]
            +
            track_names[
                track_codes[i]
            ]
            +
            frame_names[
                frame_codes[i]
            ]
            +
            ".jpg"
        )

        reconstructed.append(
            filename
        )


    print(
        "\nFirst 10 reconstructed names:"
    )

    for i in range(10):

        print(
            f"{i:2d}: "
            f"{reconstructed[i]}"
        )


    reconstructed_set = set(
        reconstructed
    )

    official_set = set(
        official
    )


    print("\n" + "-" * 100)

    print(
        "Unique reconstructed:",
        f"{len(reconstructed_set):,}"
    )

    print(
        "Unique official:",
        f"{len(official_set):,}"
    )


    duplicate_reconstructed = (
        len(reconstructed)
        -
        len(reconstructed_set)
    )


    print(
        "Duplicate reconstructed names:",
        f"{duplicate_reconstructed:,}"
    )


    shared = (
        reconstructed_set
        &
        official_set
    )

    missing = (
        official_set
        -
        reconstructed_set
    )

    extra = (
        reconstructed_set
        -
        official_set
    )


    print(
        "Shared names:",
        f"{len(shared):,}"
    )

    print(
        "Official names missing from Deep Lake reconstruction:",
        f"{len(missing):,}"
    )

    print(
        "Reconstructed names not in official list:",
        f"{len(extra):,}"
    )


    print(
        "\nExact SET match:",
        reconstructed_set
        == official_set
    )


    # --------------------------------------------------------
    # Order check
    # --------------------------------------------------------

    same_position = sum(
        a == b
        for a, b in zip(
            reconstructed,
            official
        )
    )


    print(
        "\nSame filename at same array position:",
        f"{same_position:,}"
    )

    print(
        "Same-position percentage:",
        f"{100 * same_position / len(official):.4f}%"
    )


    if missing:

        print(
            "\nFirst 20 missing official names:"
        )

        for x in sorted(
            missing
        )[:20]:

            print(" ", x)


    if extra:

        print(
            "\nFirst 20 extra reconstructed names:"
        )

        for x in sorted(
            extra
        )[:20]:

            print(" ", x)


    print("\nRESULT:")


    if (
        reconstructed_set
        == official_set
        and duplicate_reconstructed == 0
    ):

        print(
            "PASS ✅"
        )

        print(
            "Deep Lake contains the complete "
            "official MARS split."
        )

        print(
            "Sample order differs, but each image "
            "can be assigned its correct official "
            "filename using Deep Lake metadata."
        )

        return True


    print(
        "FAIL ❌"
    )

    print(
        "Do NOT export images until the "
        "mapping discrepancy is resolved."
    )

    return False


def main():

    results = []

    for cfg in SPLITS:

        results.append(
            audit_split(cfg)
        )


    print("\n" + "=" * 100)
    print("FINAL MAPPING STATUS")
    print("=" * 100)


    if all(results):

        print(
            "TRAIN: PASS"
        )

        print(
            "TEST : PASS"
        )

        print(
            "\nSafe to build corrected "
            "MARS exporter."
        )

    else:

        print(
            "At least one split failed."
        )

        print(
            "Do NOT export MARS yet."
        )


if __name__ == "__main__":
    main()
