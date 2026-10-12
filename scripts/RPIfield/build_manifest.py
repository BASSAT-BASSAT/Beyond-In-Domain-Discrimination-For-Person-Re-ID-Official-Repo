from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat


BASE = Path.home() / "DG_ReID_Audit"

ROOT = (
    BASE
    / "datasets/RPIfield/extracted/RPIfield_v0"
)

DATA = ROOT / "Data"
INFO = ROOT / "Data_Info"

OUT_DIR = (
    BASE
    / "manifests/RPIfield"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MANIFEST_OUT = (
    OUT_DIR
    / "rpifield_manifest.csv"
)

EXTRA_FOLDERS_OUT = (
    OUT_DIR
    / "rpifield_extra_folders.csv"
)

MISSING_PATHS_OUT = (
    OUT_DIR
    / "rpifield_missing_paths.csv"
)


def to_string(x):

    while (
        isinstance(x, np.ndarray)
        and x.size == 1
    ):
        x = x.item()

    if isinstance(x, bytes):
        return x.decode("utf-8")

    if isinstance(x, str):
        return x

    if (
        isinstance(x, np.ndarray)
        and x.dtype.kind in {"U", "S"}
    ):
        return "".join(
            str(v)
            for v in x.flatten()
        )

    return str(x)


rows = []
missing_paths = []
extra_folders = []

timestamp_frame_mismatches = 0
folder_label_mismatches = 0

print("=" * 90)
print("RPIFIELD MANIFEST BUILD")
print("=" * 90)


for cam in range(1, 13):

    print(f"\nProcessing Camera {cam}...")

    mat_path = (
        INFO
        / f"Cam_{cam}.mat"
    )

    mat = loadmat(
        mat_path,
        squeeze_me=True,
        chars_as_strings=True,
    )

    ids_raw = np.atleast_1d(
        mat["ids"]
    ).flatten()

    names_raw = np.atleast_1d(
        mat["allimagenames"]
    ).flatten()

    timestamps_raw = np.atleast_1d(
        mat["timestamps"]
    ).flatten()


    ids = [
        to_string(x)
        for x in ids_raw
    ]

    names = [
        to_string(x)
        for x in names_raw
    ]

    timestamps = [
        int(x)
        for x in timestamps_raw
    ]


    assert (
        len(ids)
        == len(names)
        == len(timestamps)
    )


    metadata_labels = set(
        ids
    )


    # --------------------------------------------------------
    # Filesystem folders
    # --------------------------------------------------------

    camera_dir = (
        DATA
        / f"Cam_{cam}"
    )

    filesystem_folders = {
        p.name
        for p in camera_dir.iterdir()
        if p.is_dir()
    }


    extras = (
        filesystem_folders
        - metadata_labels
    )

    for folder in sorted(
        extras
    ):

        folder_path = (
            camera_dir
            / folder
        )

        png_count = len(
            list(
                folder_path.glob(
                    "*.png"
                )
            )
        )

        extra_folders.append(
            {
                "camera_id": cam,
                "folder": folder,
                "png_count": png_count,
                "path": str(
                    folder_path
                ),
            }
        )


    # --------------------------------------------------------
    # Metadata rows
    # --------------------------------------------------------

    for label, image_name_raw, timestamp in zip(
        ids,
        names,
        timestamps,
    ):

        # MATLAB paths use backslashes
        normalized_name = (
            image_name_raw
            .replace("\\", "/")
        )

        parts = normalized_name.split(
            "/"
        )

        folder_label = parts[0]

        filename = parts[-1]


        if folder_label != label:
            folder_label_mismatches += 1


        # ----------------------------------------------------
        # Parse identity/reappearance
        # ----------------------------------------------------

        if "_" in label:

            base, reapp = label.split(
                "_",
                1
            )

            person_id = int(
                base
            )

            reappearance_index = int(
                reapp
            )

        else:

            person_id = int(
                label
            )

            reappearance_index = 0


        # ----------------------------------------------------
        # Parse frame number
        # ----------------------------------------------------

        match = re.fullmatch(
            rf"Cam_{cam}f_(\d+)\.png",
            filename,
            flags=re.IGNORECASE,
        )

        if match is None:

            raise ValueError(
                f"Unexpected filename: "
                f"{filename}"
            )


        frame_id = int(
            match.group(1)
        )


        if frame_id != timestamp:
            timestamp_frame_mismatches += 1


        # ----------------------------------------------------
        # Resolve path
        # ----------------------------------------------------

        image_path = (
            camera_dir
            / folder_label
            / filename
        )


        exists = image_path.is_file()

        if not exists:

            missing_paths.append(
                {
                    "camera_id": cam,
                    "appearance_label":
                        label,
                    "filename":
                        filename,
                    "expected_path":
                        str(image_path),
                }
            )


        # ----------------------------------------------------
        # Semantic classes
        # ----------------------------------------------------

        known_actor = (
            1
            <= person_id
            <= 112
        )

        distractor = (
            person_id
            >= 10000
        )


        sequence_id = (
            f"cam{cam:02d}:"
            f"{label}"
        )


        rows.append(
            {
                "dataset":
                    "RPIfield",

                "camera_id":
                    cam,

                "camera":
                    f"Cam_{cam}",

                "person_id":
                    person_id,

                "identity_type":
                    (
                        "known_actor"
                        if known_actor
                        else "distractor"
                    ),

                "is_known_actor":
                    known_actor,

                "is_distractor":
                    distractor,

                "appearance_label":
                    label,

                "reappearance_index":
                    reappearance_index,

                "sequence_id":
                    sequence_id,

                "frame_id":
                    frame_id,

                "timestamp":
                    timestamp,

                "filename":
                    filename,

                "relative_path":
                    str(
                        image_path.relative_to(
                            ROOT
                        )
                    ),

                "path":
                    str(
                        image_path
                    ),

                "exists":
                    exists,
            }
        )


manifest = pd.DataFrame(
    rows
)

manifest.to_csv(
    MANIFEST_OUT,
    index=False
)


pd.DataFrame(
    extra_folders
).to_csv(
    EXTRA_FOLDERS_OUT,
    index=False
)


pd.DataFrame(
    missing_paths
).to_csv(
    MISSING_PATHS_OUT,
    index=False
)


print("\n" + "=" * 90)
print("GLOBAL MANIFEST SUMMARY")
print("=" * 90)

print(
    "Rows:",
    len(manifest)
)

print(
    "Unique paths:",
    manifest[
        "path"
    ].nunique()
)

print(
    "Existing paths:",
    int(
        manifest[
            "exists"
        ].sum()
    )
)

print(
    "Missing paths:",
    len(
        missing_paths
    )
)

print(
    "Folder-label mismatches:",
    folder_label_mismatches
)

print(
    "Timestamp != frame_id:",
    timestamp_frame_mismatches
)


print("\nIDENTITIES")
print("-" * 90)

print(
    "Unique person IDs:",
    manifest[
        "person_id"
    ].nunique()
)

print(
    "Known actors:",
    manifest.loc[
        manifest[
            "is_known_actor"
        ],
        "person_id",
    ].nunique()
)

print(
    "Distractors:",
    manifest.loc[
        manifest[
            "is_distractor"
        ],
        "person_id",
    ].nunique()
)


print("\nSEQUENCES")
print("-" * 90)

print(
    "Unique sequences:",
    manifest[
        "sequence_id"
    ].nunique()
)

print(
    "Appearance labels "
    "(camera-specific):",
    manifest[
        [
            "camera_id",
            "appearance_label",
        ]
    ]
    .drop_duplicates()
    .shape[0]
)


print("\nCAMERA COVERAGE PER ID")
print("-" * 90)

camera_coverage = (
    manifest[
        [
            "person_id",
            "camera_id",
        ]
    ]
    .drop_duplicates()
    .groupby(
        "person_id"
    )[
        "camera_id"
    ]
    .nunique()
)

print(
    camera_coverage
    .describe()
    .round(2)
)


print("\nKNOWN ACTOR CAMERA COVERAGE")
print("-" * 90)

known = manifest[
    manifest[
        "is_known_actor"
    ]
]

known_camera_coverage = (
    known[
        [
            "person_id",
            "camera_id",
        ]
    ]
    .drop_duplicates()
    .groupby(
        "person_id"
    )[
        "camera_id"
    ]
    .nunique()
)

print(
    known_camera_coverage
    .describe()
    .round(2)
)

print(
    "\nKnown actors by number of cameras:"
)

print(
    known_camera_coverage
    .value_counts()
    .sort_index()
)


print("\nDISTRACTOR CAMERA COVERAGE")
print("-" * 90)

dist = manifest[
    manifest[
        "is_distractor"
    ]
]

dist_camera_coverage = (
    dist[
        [
            "person_id",
            "camera_id",
        ]
    ]
    .drop_duplicates()
    .groupby(
        "person_id"
    )[
        "camera_id"
    ]
    .nunique()
)

print(
    dist_camera_coverage
    .describe()
    .round(2)
)

print(
    "\nDistractors by number of cameras:"
)

print(
    dist_camera_coverage
    .value_counts()
    .sort_index()
)


print("\nSEQUENCE SIZE")
print("-" * 90)

sequence_sizes = (
    manifest
    .groupby(
        "sequence_id"
    )
    .size()
)

print(
    sequence_sizes
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
    .round(2)
)


print("\nEXTRA FILESYSTEM FOLDERS")
print("-" * 90)

if len(extra_folders) == 0:

    print("None")

else:

    print(
        pd.DataFrame(
            extra_folders
        ).to_string(
            index=False
        )
    )


print("\nSaved:")
print(MANIFEST_OUT)
print(EXTRA_FOLDERS_OUT)
print(MISSING_PATHS_OUT)

print("\nDONE")
