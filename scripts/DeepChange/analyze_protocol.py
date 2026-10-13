from pathlib import Path
import pandas as pd

manifest = (
    Path.home()
    / "DG_ReID_Audit/manifests/DeepChange/deepchange_manifest.csv"
)

df = pd.read_csv(manifest)

train = df[df["split"] == "train"]
vq = df[df["split"] == "val_query"]
vg = df[df["split"] == "val_gallery"]
tq = df[df["split"] == "test_query"]
tg = df[df["split"] == "test_gallery"]


def pid_set(x):
    return set(x["pid"].unique())


def tracklet_set(x):
    return set(x["tracklet_id"].unique())


train_pids = pid_set(train)
val_q_pids = pid_set(vq)
val_g_pids = pid_set(vg)
test_q_pids = pid_set(tq)
test_g_pids = pid_set(tg)

print("=" * 80)
print("IDENTITY SPLIT INTEGRITY")
print("=" * 80)

print("Train ∩ Val Query    :", len(train_pids & val_q_pids))
print("Train ∩ Val Gallery  :", len(train_pids & val_g_pids))
print("Train ∩ Test Query   :", len(train_pids & test_q_pids))
print("Train ∩ Test Gallery :", len(train_pids & test_g_pids))

val_all = val_q_pids | val_g_pids
test_all = test_q_pids | test_g_pids

print("Val ∩ Test           :", len(val_all & test_all))


print("\n" + "=" * 80)
print("QUERY / GALLERY PID COVERAGE")
print("=" * 80)

print("VAL:")
print("Query IDs            :", len(val_q_pids))
print("Gallery IDs          :", len(val_g_pids))
print("Query IDs missing G  :", len(val_q_pids - val_g_pids))
print("Gallery-only IDs     :", len(val_g_pids - val_q_pids))

print("\nTEST:")
print("Query IDs            :", len(test_q_pids))
print("Gallery IDs          :", len(test_g_pids))
print("Query IDs missing G  :", len(test_q_pids - test_g_pids))
print("Gallery-only IDs     :", len(test_g_pids - test_q_pids))


print("\n" + "=" * 80)
print("QUERY / GALLERY TRACKLET OVERLAP")
print("=" * 80)

for name, q, g in [
    ("VAL", vq, vg),
    ("TEST", tq, tg),
]:
    q_tracks = tracklet_set(q)
    g_tracks = tracklet_set(g)

    print(f"\n{name}:")
    print("Query tracklets      :", len(q_tracks))
    print("Gallery tracklets    :", len(g_tracks))
    print("Shared tracklets     :", len(q_tracks & g_tracks))
    print("Union tracklets      :", len(q_tracks | g_tracks))
    print("Query-only tracklets :", len(q_tracks - g_tracks))
    print("Gallery-only         :", len(g_tracks - q_tracks))


print("\n" + "=" * 80)
print("QUERY POSITIVE CHECK")
print("=" * 80)

for name, q, g in [
    ("VAL", vq, vg),
    ("TEST", tq, tg),
]:

    gallery_counts = g.groupby("pid").size()

    positives = q["pid"].map(gallery_counts).fillna(0)

    print(f"\n{name}:")
    print("Queries              :", len(q))
    print("Zero-positive queries:", int((positives == 0).sum()))
    print("Min positives/query  :", int(positives.min()))
    print("Median positives     :", float(positives.median()))
    print("Mean positives       :", round(float(positives.mean()), 2))
    print("Max positives        :", int(positives.max()))


print("\n" + "=" * 80)
print("CAMERA COVERAGE")
print("=" * 80)

all_cams = set(df["camera_id"].unique())
train_cams = set(train["camera_id"].unique())

print("All cameras :", sorted(all_cams))
print("Train cams  :", sorted(train_cams))
print(
    "Missing from train:",
    sorted(all_cams - train_cams)
)
