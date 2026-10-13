from pathlib import Path
from collections import Counter

import numpy as np
from scipy.io import loadmat


BASE = Path.home() / "DG_ReID_Audit"

ROOT = (
    BASE
    / "datasets/RPIfield/extracted/RPIfield_v0"
)

DATA = ROOT / "Data"
INFO = ROOT / "Data_Info"


def to_string(x):

    while isinstance(x, np.ndarray) and x.size == 1:
        x = x.item()

    if isinstance(x, bytes):
        return x.decode("utf-8")

    if isinstance(x, str):
        return x

    if isinstance(x, np.ndarray):

        if x.dtype.kind in {"U", "S"}:
            return "".join(
                str(v)
                for v in x.flatten()
            )

    return str(x)


print("=" * 90)
print("RPIFIELD METADATA INSPECTION")
print("=" * 90)


all_person_ids = set()

known_ids = set()
distractor_ids = set()
other_ids = set()

total_metadata_rows = 0


for cam in range(1, 13):

    mat_path = (
        INFO
        / f"Cam_{cam}.mat"
    )

    data = loadmat(
        mat_path,
        squeeze_me=True,
        chars_as_strings=True,
    )


    print("\n" + "=" * 90)
    print(f"CAMERA {cam}")
    print("=" * 90)


    public_keys = [
        k
        for k in data.keys()
        if not k.startswith("__")
    ]

    print(
        "MAT keys:",
        public_keys
    )


    ids_raw = np.atleast_1d(
        data["ids"]
    ).flatten()

    names_raw = np.atleast_1d(
        data["allimagenames"]
    ).flatten()

    timestamps_raw = np.atleast_1d(
        data["timestamps"]
    ).flatten()


    ids = [
        to_string(x)
        for x in ids_raw
    ]

    names = [
        to_string(x)
        for x in names_raw
    ]


    filesystem_images = list(
        (
            DATA
            / f"Cam_{cam}"
        ).rglob("*.png")
    )


    print(
        "Metadata rows:",
        len(ids)
    )

    print(
        "Image names:",
        len(names)
    )

    print(
        "Timestamps:",
        len(timestamps_raw)
    )

    print(
        "Filesystem images:",
        len(filesystem_images)
    )


    print(
        "Counts all equal:",
        len(ids)
        == len(names)
        == len(timestamps_raw)
        == len(filesystem_images)
    )


    # ----------------------------------------
    # Parse base PID + reappearance
    # ----------------------------------------

    base_ids = []

    reapp_indices = []


    for token in ids:

        if "_" in token:

            base, reapp = token.split(
                "_",
                1
            )

            pid = int(base)
            reapp_index = int(reapp)

        else:

            pid = int(token)
            reapp_index = 0


        base_ids.append(
            pid
        )

        reapp_indices.append(
            reapp_index
        )

        all_person_ids.add(
            pid
        )

        if 1 <= pid <= 112:

            known_ids.add(
                pid
            )

        elif pid >= 10000:

            distractor_ids.add(
                pid
            )

        else:

            other_ids.add(
                pid
            )


    print(
        "Unique folder/appearance labels:",
        len(set(ids))
    )

    print(
        "Unique base person IDs:",
        len(set(base_ids))
    )


    known_cam = {
        x
        for x in base_ids
        if 1 <= x <= 112
    }

    distractor_cam = {
        x
        for x in base_ids
        if x >= 10000
    }

    other_cam = {
        x
        for x in base_ids
        if not (
            1 <= x <= 112
            or x >= 10000
        )
    }


    print(
        "Known actor IDs:",
        len(known_cam)
    )

    print(
        "Distractor IDs:",
        len(distractor_cam)
    )

    print(
        "Other/unexpected IDs:",
        len(other_cam)
    )


    print(
        "Max reappearance index:",
        max(reapp_indices)
    )


    # ----------------------------------------
    # Timestamp inspection
    # ----------------------------------------

    print("\nTimestamp type:")

    print(
        "dtype:",
        timestamps_raw.dtype
    )


    print(
        "First 10 raw timestamps:"
    )

    print(
        timestamps_raw[:10]
    )


    try:

        ts_numeric = np.asarray(
            timestamps_raw,
            dtype=float
        )

        print(
            "Timestamp min:",
            float(
                np.min(ts_numeric)
            )
        )

        print(
            "Timestamp max:",
            float(
                np.max(ts_numeric)
            )
        )

        print(
            "Timestamp unique:",
            len(
                np.unique(
                    ts_numeric
                )
            )
        )

    except Exception as e:

        print(
            "Could not convert timestamps to float:",
            e
        )


    # ----------------------------------------
    # Examples
    # ----------------------------------------

    print("\nFirst 10 records:")

    for i in range(
        min(
            10,
            len(ids)
        )
    ):

        print(
            f"{i:>3} | "
            f"id={ids[i]:<12} | "
            f"name={names[i]:<30} | "
            f"timestamp={timestamps_raw[i]}"
        )


    total_metadata_rows += len(ids)


print("\n" + "=" * 90)
print("GLOBAL SUMMARY")
print("=" * 90)

print(
    "Total metadata rows:",
    total_metadata_rows
)

print(
    "Total PNG expected:",
    601581
)

print(
    "Global unique base person IDs:",
    len(all_person_ids)
)

print(
    "Known actor IDs observed:",
    len(known_ids)
)

print(
    "Distractor IDs observed:",
    len(distractor_ids)
)

print(
    "Unexpected IDs observed:",
    len(other_ids)
)


if other_ids:

    print(
        "Unexpected ID examples:",
        sorted(
            other_ids
        )[:50]
    )


print("\nDONE")
