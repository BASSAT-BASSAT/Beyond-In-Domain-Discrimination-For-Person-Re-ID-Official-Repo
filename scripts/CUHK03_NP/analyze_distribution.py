from pathlib import Path
import pandas as pd

MANIFEST_DIR = Path.home() / "DG_ReID_Audit/manifests/CUHK03_NP"
REPORT_DIR = Path.home() / "DG_ReID_Audit/reports/CUHK03_NP"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

for variant in ["detected", "labeled"]:

    df = pd.read_csv(MANIFEST_DIR / f"{variant}_manifest.csv")

    print("\n" + "=" * 70)
    print(variant.upper())
    print("=" * 70)

    for split in ["train", "gallery"]:

        sdf = df[df["split"] == split].copy()

        print(f"\n---------------- {split.upper()} ----------------")

        # عدد الصور لكل Identity
        images_per_pid = sdf.groupby("pid").size()

        print("\nImages per ID:")
        print(f"IDs    : {len(images_per_pid)}")
        print(f"Min    : {images_per_pid.min()}")
        print(f"Max    : {images_per_pid.max()}")
        print(f"Mean   : {images_per_pid.mean():.2f}")
        print(f"Median : {images_per_pid.median():.2f}")
        print(f"Std    : {images_per_pid.std():.2f}")

        # عدد الصور لكل View
        print("\nImages per view:")
        print(sdf["view"].value_counts().sort_index())

        # كل PID ظهر في كام view
        views_per_pid = sdf.groupby("pid")["view"].nunique()

        print("\nNumber of views per ID:")
        print(views_per_pid.value_counts().sort_index())

        # جدول per-identity
        pivot = (
            sdf.groupby(["pid", "view"])
               .size()
               .unstack(fill_value=0)
        )

        for view in [1, 2]:
            if view not in pivot.columns:
                pivot[view] = 0

        pivot = pivot[[1, 2]]
        pivot.columns = ["view1_images", "view2_images"]

        pivot["total_images"] = (
            pivot["view1_images"] +
            pivot["view2_images"]
        )

        pivot["view_difference"] = (
            pivot["view1_images"] -
            pivot["view2_images"]
        ).abs()

        # IDs ناقصة view
        missing_view1 = (pivot["view1_images"] == 0).sum()
        missing_view2 = (pivot["view2_images"] == 0).sum()

        print("\nMissing views:")
        print("IDs missing view 1:", missing_view1)
        print("IDs missing view 2:", missing_view2)

        print("\nMost imbalanced IDs between views:")
        print(
            pivot.sort_values(
                "view_difference",
                ascending=False
            ).head(10)
        )

        output_file = (
            REPORT_DIR /
            f"{variant}_{split}_pid_distribution.csv"
        )

        pivot.to_csv(output_file)

        print("\nSaved:", output_file)

    print("\n")
