from pathlib import Path
import pandas as pd

MANIFEST = (
    Path.home()
    / "DG_ReID_Audit/manifests/IUST_PersonReID/iust_manifest_raw.csv"
)

REPORT_DIR = (
    Path.home()
    / "DG_ReID_Audit/reports/IUST_PersonReID"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)

query = df[df["split"] == "query"].copy()
test = df[df["split"] == "test"].copy()

print("=" * 70)
print("IUST QUERY / TEST PROTOCOL AUDIT")
print("=" * 70)

# ---------------------------------------------------
# 1. Query count per PID-camera
# ---------------------------------------------------

query_pid_cam = (
    query.groupby(["pid", "camera"])
    .size()
)

print("\nQuery images per (PID, camera):")
print(
    query_pid_cam
    .value_counts()
    .sort_index()
)

print(
    "\n(PID, camera) pairs with more than one query:",
    (query_pid_cam > 1).sum()
)

# ---------------------------------------------------
# 2. Does query cover every camera of each test PID?
# ---------------------------------------------------

test_camera_sets = (
    test.groupby("pid")["camera"]
    .apply(lambda x: set(x))
)

query_camera_sets = (
    query.groupby("pid")["camera"]
    .apply(lambda x: set(x))
)

mismatch = []

for pid in sorted(query["pid"].unique()):

    test_cams = test_camera_sets[pid]
    query_cams = query_camera_sets[pid]

    if test_cams != query_cams:
        mismatch.append({
            "pid": pid,
            "test_cameras": sorted(test_cams),
            "query_cameras": sorted(query_cams),
        })

print(
    "\nPIDs where Query camera set != Test camera set:",
    len(mismatch)
)

if mismatch:
    print("\nExamples:")
    for item in mismatch[:10]:
        print(item)

# ---------------------------------------------------
# 3. Evaluate valid positives for each query
# ---------------------------------------------------

rows = []

for _, q in query.iterrows():

    same_pid = test[
        test["pid"] == q["pid"]
    ]

    exact_same_file = same_pid[
        same_pid["filename"] == q["filename"]
    ]

    same_camera = same_pid[
        same_pid["camera"] == q["camera"]
    ]

    cross_camera = same_pid[
        same_pid["camera"] != q["camera"]
    ]

    rows.append({
        "query_filename": q["filename"],
        "pid": q["pid"],
        "query_camera": q["camera"],

        "all_same_pid_test_images": len(same_pid),

        "exact_query_copy_in_test": len(exact_same_file),

        "same_camera_same_pid": len(same_camera),

        "cross_camera_positives": len(cross_camera),

        "num_other_cameras": cross_camera["camera"].nunique(),
    })

qstats = pd.DataFrame(rows)

print("\n" + "=" * 70)
print("VALID POSITIVES")
print("=" * 70)

print("\nQueries checked:", len(qstats))

print(
    "Queries with exact copy in test:",
    (qstats["exact_query_copy_in_test"] > 0).sum()
)

print(
    "Queries with ZERO cross-camera positives:",
    (qstats["cross_camera_positives"] == 0).sum()
)

print("\nCross-camera positives per query:")
print(
    qstats["cross_camera_positives"]
    .describe()
    .round(2)
)

print("\nOther cameras available per query:")
print(
    qstats["num_other_cameras"]
    .value_counts()
    .sort_index()
)

print("\nQueries with fewest cross-camera positives:")

print(
    qstats.sort_values(
        "cross_camera_positives"
    )
    .head(15)
    .to_string(index=False)
)

output = REPORT_DIR / "query_test_protocol.csv"

qstats.to_csv(
    output,
    index=False
)

print("\nSaved:", output)
