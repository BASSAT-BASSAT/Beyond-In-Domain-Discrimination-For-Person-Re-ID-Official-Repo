from pathlib import Path
from collections import Counter, defaultdict


ROOT = (
    Path.home()
    / "DG_ReID_Audit/datasets/LPW/extracted/pep_256x128"
)


print("=" * 90)
print("LPW STRUCTURE AUDIT")
print("=" * 90)


scenes = sorted(
    [
        p for p in ROOT.iterdir()
        if p.is_dir()
    ]
)


total_images = 0
total_tracklets = 0

global_identities = set()

scene_stats = {}


for scene_dir in scenes:

    scene = scene_dir.name

    views = sorted(
        [
            p for p in scene_dir.iterdir()
            if p.is_dir()
        ]
    )

    scene_ids = set()

    scene_images = 0
    scene_tracklets = 0

    view_stats = {}


    for view_dir in views:

        view = view_dir.name

        person_dirs = sorted(
            [
                p for p in view_dir.iterdir()
                if p.is_dir()
            ]
        )

        view_images = 0


        for person_dir in person_dirs:

            person_local_id = person_dir.name

            global_pid = (
                scene,
                person_local_id
            )

            scene_ids.add(
                global_pid
            )

            global_identities.add(
                global_pid
            )


            images = list(
                person_dir.glob("*.jpg")
            )

            n_images = len(images)

            view_images += n_images
            scene_images += n_images
            total_images += n_images


            # Each (scene, view, person folder)
            # represents one tracklet / sequence.
            scene_tracklets += 1
            total_tracklets += 1


        view_stats[view] = {
            "person_folders":
                len(person_dirs),

            "images":
                view_images,
        }


    scene_stats[scene] = {
        "views":
            len(views),

        "identities":
            len(scene_ids),

        "tracklets":
            scene_tracklets,

        "images":
            scene_images,

        "view_stats":
            view_stats,
    }


# ============================================================
# Report
# ============================================================

print("\nSCENES")
print("-" * 90)

for scene, stats in scene_stats.items():

    print(f"\n{scene.upper()}")

    print(
        "Views      :",
        stats["views"]
    )

    print(
        "Identities :",
        stats["identities"]
    )

    print(
        "Tracklets  :",
        stats["tracklets"]
    )

    print(
        "Images     :",
        stats["images"]
    )


    print("\nPer view:")

    for view, vstats in (
        stats["view_stats"].items()
    ):

        print(
            f"  {view:<8}"
            f" persons={vstats['person_folders']:<5}"
            f" images={vstats['images']}"
        )


print("\n" + "=" * 90)
print("GLOBAL TOTALS")
print("=" * 90)

print(
    "Scenes      :",
    len(scenes)
)

print(
    "Identities  :",
    len(global_identities)
)

print(
    "Tracklets   :",
    total_tracklets
)

print(
    "Images      :",
    total_images
)


# ============================================================
# Camera/view coverage per identity
# ============================================================

id_views = defaultdict(set)

id_images = Counter()


for scene_dir in scenes:

    scene = scene_dir.name

    for view_dir in scene_dir.iterdir():

        if not view_dir.is_dir():
            continue

        view = view_dir.name

        for person_dir in view_dir.iterdir():

            if not person_dir.is_dir():
                continue

            pid = (
                scene,
                person_dir.name
            )

            id_views[pid].add(
                view
            )

            id_images[pid] += len(
                list(
                    person_dir.glob(
                        "*.jpg"
                    )
                )
            )


coverage = Counter(
    len(views)
    for views in id_views.values()
)


print("\n" + "=" * 90)
print("VIEW COVERAGE PER IDENTITY")
print("=" * 90)

for n_views, count in sorted(
    coverage.items()
):

    print(
        f"{n_views} view(s): "
        f"{count} identities"
    )


# ============================================================
# Images per identity
# ============================================================

values = sorted(
    id_images.values()
)


print("\n" + "=" * 90)
print("IMAGES PER IDENTITY")
print("=" * 90)

print(
    "Min    :",
    min(values)
)

print(
    "Median :",
    values[
        len(values) // 2
    ]
)

print(
    "Mean   :",
    round(
        sum(values)
        / len(values),
        2
    )
)

print(
    "Max    :",
    max(values)
)


# ============================================================
# Official split
# ============================================================

train_ids = {
    pid
    for pid in global_identities
    if pid[0] in {
        "scen2",
        "scen3"
    }
}

test_ids = {
    pid
    for pid in global_identities
    if pid[0] == "scen1"
}


print("\n" + "=" * 90)
print("OFFICIAL SCENE SPLIT")
print("=" * 90)

print(
    "Train identities "
    "(scen2 + scen3):",
    len(train_ids)
)

print(
    "Test identities "
    "(scen1):",
    len(test_ids)
)

print(
    "Train/Test overlap:",
    len(
        train_ids
        & test_ids
    )
)


print("\n" + "=" * 90)
print("EXPECTED REFERENCE VALUES")
print("=" * 90)

print("Images     : 590547")
print("Identities : 2731")
print("Tracklets  : 7694")
print("Train IDs  : 1975")
print("Test IDs   : 756")


print("\nDONE")
