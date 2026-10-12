from pathlib import Path
from urllib.request import urlretrieve
import time

import deeplake
import numpy as np
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

ROOT = (
    Path.home()
    / "DG_ReID_Audit"
    / "datasets"
    / "MARS"
)

INFO_DIR = ROOT / "info"

TRAIN_DATASET = "hub://activeloop/mars-train"
TEST_DATASET = "hub://activeloop/mars-test"

EXPECTED_TRAIN = 509_914
EXPECTED_TEST = 681_089

INFO_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "liangzheng06/MARS-evaluation/master/info"
)

INFO_FILES = [
    "train_name.txt",
    "test_name.txt",
    "tracks_train_info.mat",
    "tracks_test_info.mat",
    "query_IDX.mat",
]


# ============================================================
# Metadata
# ============================================================

def download_metadata():

    INFO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 90)
    print("OFFICIAL MARS METADATA")
    print("=" * 90)

    for filename in INFO_FILES:

        output = INFO_DIR / filename

        if output.exists():
            print(f"[SKIP] {filename}")
            continue

        url = (
            f"{INFO_BASE_URL}/"
            f"{filename}"
        )

        print(f"[DOWNLOAD] {filename}")

        urlretrieve(
            url,
            output,
        )


# ============================================================
# Helpers
# ============================================================

def get_codes(ds, tensor_name):

    values = (
        ds[tensor_name][:]
        .numpy()
        .reshape(-1)
    )

    return values.astype(
        np.int64
    )


def get_class_names(ds, tensor_name):

    info = ds[tensor_name].info

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
        f"Could not read class names: "
        f"{tensor_name}"
    )


def count_images(folder):

    if not folder.exists():
        return 0

    return sum(
        1
        for path in folder.rglob("*.jpg")
        if path.is_file()
    )


# ============================================================
# Export
# ============================================================

def export_split(
    dataset_url,
    output_dir,
    expected_count,
):

    print("\n" + "=" * 90)
    print(f"LOADING {dataset_url}")
    print("=" * 90)

    ds = deeplake.load(
        dataset_url,
        read_only=True,
    )

    total = len(ds)

    print(
        f"\nSamples: {total:,}"
    )

    print(
        f"Expected: {expected_count:,}"
    )

    if total != expected_count:

        raise RuntimeError(
            "Dataset size mismatch.\n"
            f"Found:    {total:,}\n"
            f"Expected: {expected_count:,}"
        )

    required = [
        "images",
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
        "\nReading class-label metadata..."
    )

    pid_codes = get_codes(
        ds,
        "pedestrian_ids",
    )

    camera_codes = get_codes(
        ds,
        "camera_nos",
    )

    track_codes = get_codes(
        ds,
        "track_nos",
    )

    frame_codes = get_codes(
        ds,
        "frame_nos",
    )

    pid_names = get_class_names(
        ds,
        "pedestrian_ids",
    )

    camera_names = get_class_names(
        ds,
        "camera_nos",
    )

    track_names = get_class_names(
        ds,
        "track_nos",
    )

    frame_names = get_class_names(
        ds,
        "frame_nos",
    )

    print(
        "Metadata loaded."
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Cache PID directories so we don't mkdir
    # hundreds of thousands of times.
    created_dirs = set()

    written = 0
    skipped = 0

    print("\n" + "=" * 90)
    print("EXPORTING ORIGINAL JPEG BYTES")
    print("=" * 90)

    print(
        f"Destination: {output_dir}"
    )

    # Deep Lake explicitly recommends iteration
    # instead of repeated ds[i] access.
    for i, sample in enumerate(
        tqdm(
            ds,
            total=total,
            desc=output_dir.name,
        )
    ):

        pid = pid_names[
            pid_codes[i]
        ]

        camera = camera_names[
            camera_codes[i]
        ]

        track = track_names[
            track_codes[i]
        ]

        frame = frame_names[
            frame_codes[i]
        ]

        filename = (
            pid
            + camera
            + track
            + frame
            + ".jpg"
        )

        pid_dir = (
            output_dir
            / pid
        )

        if pid not in created_dirs:

            pid_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            created_dirs.add(
                pid
            )

        output_file = (
            pid_dir
            / filename
        )

        # Resume support.
        if output_file.exists():

            skipped += 1
            continue

        # ----------------------------------------------
        # IMPORTANT:
        #
        # Do NOT decode and re-save the JPEG.
        #
        # Deep Lake stores JPEG compressed bytes.
        # We write those bytes directly.
        # ----------------------------------------------

        # ----------------------------------------------
        # Robust network retry for Deep Lake / S3
        # ----------------------------------------------

        max_retries = 10
        raw = None

        for attempt in range(
            1,
            max_retries + 1,
        ):

            try:

                raw = sample[
                    "images"
                ].tobytes()

                break

            except Exception as e:

                print(
                    f"\\n[NETWORK RETRY] "
                    f"{filename}"
                )

                print(
                    f"Attempt "
                    f"{attempt}/{max_retries}"
                )

                print(
                    f"Error: "
                    f"{type(e).__name__}: {e}"
                )

                if attempt == max_retries:

                    raise RuntimeError(
                        "Failed to download "
                        f"{filename} after "
                        f"{max_retries} retries."
                    ) from e

                wait_seconds = min(
                    5 * attempt,
                    30,
                )

                print(
                    f"Waiting "
                    f"{wait_seconds}s..."
                )

                time.sleep(
                    wait_seconds
                )

        if raw is None:

            raise RuntimeError(
                f"No JPEG data returned "
                f"for {filename}"
            )

        if not raw.startswith(
            b"\\xff\\xd8\\xff"
        ):

            raise RuntimeError(
                "Expected JPEG bytes for "
                f"{filename}, but JPEG magic "
                "was not found."
            )

        # Atomic write:
        # first create temporary file,
        # then rename only after the complete
        # JPEG has been written.
        temp_file = (
            output_file.with_suffix(
                ".jpg.part"
            )
        )

        temp_file.write_bytes(
            raw
        )

        temp_file.replace(
            output_file
        )

        written += 1

    print("\nExport finished.")

    print(
        f"Written : {written:,}"
    )

    print(
        f"Skipped : {skipped:,}"
    )

    print(
        f"Total   : "
        f"{written + skipped:,}"
    )


# ============================================================
# Verification
# ============================================================

def verify():

    train_dir = ROOT / "bbox_train"
    test_dir = ROOT / "bbox_test"

    print("\n" + "=" * 90)
    print("FINAL MARS VERIFICATION")
    print("=" * 90)

    train_count = count_images(
        train_dir
    )

    test_count = count_images(
        test_dir
    )

    print(
        f"\nTrain images: "
        f"{train_count:,} / "
        f"{EXPECTED_TRAIN:,}"
    )

    print(
        f"Test images : "
        f"{test_count:,} / "
        f"{EXPECTED_TEST:,}"
    )

    print("\nMetadata:")

    metadata_ok = True

    for filename in INFO_FILES:

        path = INFO_DIR / filename

        exists = path.exists()

        if not exists:
            metadata_ok = False

        print(
            f"{filename:<25} "
            f"{'OK' if exists else 'MISSING'}"
        )

    train_ok = (
        train_count
        == EXPECTED_TRAIN
    )

    test_ok = (
        test_count
        == EXPECTED_TEST
    )

    print("\nChecks:")

    print(
        "Train:",
        "PASS" if train_ok else "FAIL"
    )

    print(
        "Test :",
        "PASS" if test_ok else "FAIL"
    )

    print(
        "Info :",
        "PASS" if metadata_ok else "FAIL"
    )

    if (
        train_ok
        and test_ok
        and metadata_ok
    ):

        print(
            "\n"
            "MARS DATASET EXPORT COMPLETE ✅"
        )

        print(
            "Original Deep Lake JPEG bytes "
            "were preserved without "
            "recompression."
        )

    else:

        print(
            "\nMARS EXPORT INCOMPLETE ❌"
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 90)
    print("MARS SAFE DATASET EXPORTER")
    print("=" * 90)

    print(
        f"\nOutput root:\n{ROOT}"
    )

    ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Official evaluation metadata
    download_metadata()

    # Train
    export_split(
        TRAIN_DATASET,
        ROOT / "bbox_train",
        EXPECTED_TRAIN,
    )

    # Test
    export_split(
        TEST_DATASET,
        ROOT / "bbox_test",
        EXPECTED_TEST,
    )

    # Final validation
    verify()


if __name__ == "__main__":
    main()
