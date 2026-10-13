from pathlib import Path
from collections import defaultdict, Counter
import re

ROOT = Path.home() / "DG_ReID_Audit/datasets/DeepChange"

FILES = {
    "train": ROOT / "train-set-bbox.txt",
    "val_query": ROOT / "val-set-query-bbox.txt",
    "val_gallery": ROOT / "val-set-gallery-bbox.txt",
    "test_query": ROOT / "test-set-query-bbox.txt",
    "test_gallery": ROOT / "test-set-gallery-bbox.txt",
}

pattern = re.compile(
    r"^p(?P<pid>\d{4})_"
    r"c(?P<camera>\d{2})_"
    r"(?P<date>\d{8})_"
    r"(?P<time>\d{6})_"
    r"(?P<frame>\d+)_bbox\.jpg$"
)

for split, file_path in FILES.items():

    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:

            line = line.strip()

            if not line:
                continue

            filename, second_field = line.rsplit(",", 1)

            m = pattern.match(filename)

            if not m:
                print("INVALID:", filename)
                continue

            rows.append({
                "pid": int(m.group("pid")),
                "camera": int(m.group("camera")),
                "date": m.group("date"),
                "time": m.group("time"),
                "frame": int(m.group("frame")),
                "second": int(second_field),
            })

    pids = {r["pid"] for r in rows}
    labels = {r["second"] for r in rows}
    cameras = {r["camera"] for r in rows}
    dates = {r["date"] for r in rows}

    pid_to_labels = defaultdict(set)
    label_to_pids = defaultdict(set)

    for r in rows:
        pid_to_labels[r["pid"]].add(r["second"])
        label_to_pids[r["second"]].add(r["pid"])

    labels_per_pid = Counter(
        len(v) for v in pid_to_labels.values()
    )

    pids_per_label = Counter(
        len(v) for v in label_to_pids.values()
    )

    print("\n" + "=" * 80)
    print(split.upper())
    print("=" * 80)

    print("Images             :", len(rows))
    print("Unique PIDs        :", len(pids))
    print("Unique second vals :", len(labels))
    print("Unique cameras     :", len(cameras))
    print("Unique dates       :", len(dates))

    print("\nSecond values per PID:")
    print(dict(sorted(labels_per_pid.items())))

    print("\nPIDs per second value:")
    print(dict(sorted(pids_per_label.items())))

    print("\nExample PID mappings:")
    for pid in sorted(pid_to_labels)[:15]:
        print(
            f"PID {pid:04d} -> "
            f"{sorted(pid_to_labels[pid])[:30]}"
        )

    shared_labels = [
        (label, sorted(ps))
        for label, ps in label_to_pids.items()
        if len(ps) > 1
    ]

    print(
        "\nSecond values shared by multiple PIDs:",
        len(shared_labels)
    )

    for label, ps in shared_labels[:10]:
        print(
            f"  value {label} -> PIDs {ps[:20]}"
        )
