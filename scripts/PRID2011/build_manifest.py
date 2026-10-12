from pathlib import Path
import pandas as pd


BASE = Path.home() / "DG_ReID_Audit"

ROOT = (
    BASE
    / "datasets/PRID2011/extracted/prid_2011"
)

OUT = (
    BASE
    / "manifests/PRID2011"
    / "prid2011_manifest.csv"
)


rows = []
invalid = []


def make_global_pid(camera, person_num):
    """
    README rule:
    persons 0001..0200 are shared across cameras.
    persons >0200 are camera-specific identities.
    """

    if 1 <= person_num <= 200:
        return f"shared-{person_num:04d}"

    return f"{camera}-{person_num:04d}"


# ============================================================
# SINGLE SHOT
# ============================================================

for camera in ["cam_a", "cam_b"]:

    folder = (
        ROOT
        / "single_shot"
        / camera
    )

    for image_path in sorted(
        folder.glob("*.png")
    ):

        stem = image_path.stem

        # expected:
        # person_0001
        if not stem.startswith("person_"):

            invalid.append(
                str(image_path)
            )

            continue

        person_text = stem.replace(
            "person_",
            ""
        )

        if not person_text.isdigit():

            invalid.append(
                str(image_path)
            )

            continue

        person_num = int(
            person_text
        )

        global_pid = make_global_pid(
            camera,
            person_num
        )

        trajectory_id = (
            f"{camera}-person_{person_num:04d}"
        )

        rows.append(
            {
                "dataset":
                    "PRID2011",

                "version":
                    "single_shot",

                "camera":
                    camera,

                "person_local_id":
                    f"person_{person_num:04d}",

                "person_number":
                    person_num,

                "global_person_id":
                    global_pid,

                "cross_camera_matched":
                    person_num <= 200,

                "trajectory_id":
                    trajectory_id,

                "sequence_index":
                    None,

                "filename":
                    image_path.name,

                "path":
                    str(image_path),
            }
        )


# ============================================================
# MULTI SHOT
# ============================================================

for camera in ["cam_a", "cam_b"]:

    camera_dir = (
        ROOT
        / "multi_shot"
        / camera
    )

    person_dirs = sorted(
        [
            p
            for p in camera_dir.iterdir()
            if p.is_dir()
        ]
    )

    for person_dir in person_dirs:

        name = person_dir.name

        if not name.startswith("person_"):

            invalid.append(
                str(person_dir)
            )

            continue

        person_text = name.replace(
            "person_",
            ""
        )

        if not person_text.isdigit():

            invalid.append(
                str(person_dir)
            )

            continue

        person_num = int(
            person_text
        )

        global_pid = make_global_pid(
            camera,
            person_num
        )

        trajectory_id = (
            f"{camera}-person_{person_num:04d}"
        )


        for image_path in sorted(
            person_dir.glob("*.png")
        ):

            stem = image_path.stem

            if stem.isdigit():
                sequence_index = int(stem)
            else:
                sequence_index = None
                invalid.append(
                    str(image_path)
                )


            rows.append(
                {
                    "dataset":
                        "PRID2011",

                    "version":
                        "multi_shot",

                    "camera":
                        camera,

                    "person_local_id":
                        f"person_{person_num:04d}",

                    "person_number":
                        person_num,

                    "global_person_id":
                        global_pid,

                    "cross_camera_matched":
                        person_num <= 200,

                    "trajectory_id":
                        trajectory_id,

                    "sequence_index":
                        sequence_index,

                    "filename":
                        image_path.name,

                    "path":
                        str(image_path),
                }
            )


df = pd.DataFrame(rows)


OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUT,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("=" * 90)
print("PRID2011 MANIFEST BUILD")
print("=" * 90)

print(
    "\nRows:",
    len(df)
)

print(
    "Invalid paths/names:",
    len(invalid)
)

print(
    "Unique paths:",
    df["path"].nunique()
)


print("\n" + "=" * 90)
print("BY VERSION")
print("=" * 90)

for version in [
    "single_shot",
    "multi_shot",
]:

    x = df[
        df["version"] == version
    ]

    print(
        f"\n{version.upper()}"
    )

    print(
        "Images:",
        len(x)
    )

    print(
        "Global real IDs:",
        x[
            "global_person_id"
        ].nunique()
    )

    print(
        "Trajectories:",
        x[
            "trajectory_id"
        ].nunique()
    )

    for camera in [
        "cam_a",
        "cam_b",
    ]:

        c = x[
            x["camera"] == camera
        ]

        print(
            f"  {camera}: "
            f"images={len(c)}, "
            f"local persons={c['person_local_id'].nunique()}"
        )


# ============================================================
# TRUE ID STRUCTURE
# ============================================================

print("\n" + "=" * 90)
print("REAL IDENTITY STRUCTURE")
print("=" * 90)

multi = df[
    df["version"] == "multi_shot"
]

shared_ids = set(
    multi.loc[
        multi[
            "cross_camera_matched"
        ],
        "global_person_id"
    ]
)

a_only_ids = set(
    multi.loc[
        (
            multi["camera"] == "cam_a"
        )
        &
        (
            ~multi[
                "cross_camera_matched"
            ]
        ),
        "global_person_id"
    ]
)

b_only_ids = set(
    multi.loc[
        (
            multi["camera"] == "cam_b"
        )
        &
        (
            ~multi[
                "cross_camera_matched"
            ]
        ),
        "global_person_id"
    ]
)


print(
    "Cross-camera matched IDs:",
    len(shared_ids)
)

print(
    "A-only IDs:",
    len(a_only_ids)
)

print(
    "B-only IDs:",
    len(b_only_ids)
)

print(
    "Total unique real IDs:",
    len(
        shared_ids
        | a_only_ids
        | b_only_ids
    )
)


# ============================================================
# MATCHED PERSON CAMERA COVERAGE
# ============================================================

matched = multi[
    multi[
        "cross_camera_matched"
    ]
]

coverage = (
    matched.groupby(
        "global_person_id"
    )["camera"]
    .nunique()
)


print("\nMATCHED ID CAMERA COVERAGE")
print("-" * 90)

print(
    coverage.value_counts()
    .sort_index()
)


# ============================================================
# MULTI-SHOT SEQUENCE COUNTS
# ============================================================

seq_sizes = (
    multi.groupby(
        "trajectory_id"
    )
    .size()
)


print("\n" + "=" * 90)
print("MULTI-SHOT IMAGES PER TRAJECTORY")
print("=" * 90)

print(
    seq_sizes
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


# ============================================================
# OFFICIAL EVALUATION LOGIC
# ============================================================

multi_a = multi[
    multi["camera"] == "cam_a"
]

multi_b = multi[
    multi["camera"] == "cam_b"
]


probe_a_ids = set(
    multi_a.loc[
        multi_a["person_number"] <= 200,
        "global_person_id"
    ]
)

probe_b_ids = set(
    multi_b.loc[
        multi_b["person_number"] <= 200,
        "global_person_id"
    ]
)

gallery_a_ids = set(
    multi_a["global_person_id"]
)

gallery_b_ids = set(
    multi_b["global_person_id"]
)


print("\n" + "=" * 90)
print("OFFICIAL EVALUATION ID LOGIC")
print("=" * 90)

print("\nA -> B")

print(
    "Probe IDs:",
    len(probe_a_ids)
)

print(
    "Gallery IDs:",
    len(gallery_b_ids)
)

print(
    "Probe IDs missing from B gallery:",
    len(
        probe_a_ids
        - gallery_b_ids
    )
)

print(
    "B gallery distractor IDs:",
    len(
        gallery_b_ids
        - probe_a_ids
    )
)


print("\nB -> A")

print(
    "Probe IDs:",
    len(probe_b_ids)
)

print(
    "Gallery IDs:",
    len(gallery_a_ids)
)

print(
    "Probe IDs missing from A gallery:",
    len(
        probe_b_ids
        - gallery_a_ids
    )
)

print(
    "A gallery distractor IDs:",
    len(
        gallery_a_ids
        - probe_b_ids
    )
)


# ============================================================
# SINGLE vs MULTI
# ============================================================

single = df[
    df["version"] == "single_shot"
]

print("\n" + "=" * 90)
print("SINGLE / MULTI ID CONSISTENCY")
print("=" * 90)

for camera in [
    "cam_a",
    "cam_b",
]:

    s_ids = set(
        single.loc[
            single["camera"] == camera,
            "person_local_id"
        ]
    )

    m_ids = set(
        multi.loc[
            multi["camera"] == camera,
            "person_local_id"
        ]
    )

    print(
        camera,
        "same local ID set:",
        s_ids == m_ids
    )


print("\nSaved:")
print(OUT)

print("\nDONE")
