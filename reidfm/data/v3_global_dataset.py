"""
Global P3/P5 to REIDPACK3 adapter.

Reads the authoritative Global CSV manifests.
Keeps source identity metadata unchanged.
Loads RGB images directly from packed archives.

No legacy Pool V2 conversion.
No FastReID dependency in this module.
"""

from __future__ import annotations

from pathlib import Path
import csv
import io
import os

from PIL import Image

from reidfm.data.v3_pack_reader import V3PackReader


class V3GlobalDataset:

    def __init__(
        self,
        manifest_path,
        pack_root,
        split,
    ):
        self.manifest_path = Path(manifest_path)
        self.pack_root = Path(pack_root)
        self.split = str(split)

        if self.split not in ("train", "validation"):
            raise ValueError("Invalid split")

        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)

        if not self.pack_root.is_dir():
            raise FileNotFoundError(self.pack_root)

        self._readers = {}
        self._process_id = os.getpid()

    def iter_records(self):
        with self.manifest_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            reader = csv.DictReader(stream)

            required = {
                "sample_uid",
                "dataset_root_id",
                "relative_path",
                "identity_key",
            }

            if self.split == "train":
                required.add("class_id")
            else:
                required.update({
                    "validation_protocol_id",
                    "retrieval_role",
                })

            missing = required - set(reader.fieldnames or [])

            if missing:
                raise ValueError(
                    f"Missing manifest fields: {sorted(missing)}"
                )

            for row in reader:
                if not row["dataset_root_id"]:
                    raise ValueError("Missing dataset root")

                if not row["relative_path"]:
                    raise ValueError("Missing image path")

                if not row["identity_key"]:
                    raise ValueError("Missing identity key")

                if self.split == "train":
                    value = int(row["class_id"])
                    if value < 0:
                        raise ValueError("Negative training class ID")
                else:
                    if row["retrieval_role"] not in (
                        "query", "gallery"
                    ):
                        raise ValueError("Invalid retrieval role")

                yield row

    def _reader(self, dataset_root_id):
        # Prevent sharing file/SQLite handles across
        # DataLoader processes.
        current_pid = os.getpid()

        if current_pid != self._process_id:
            self._readers = {}
            self._process_id = current_pid

        if dataset_root_id not in self._readers:
            self._readers[dataset_root_id] = V3PackReader(
                self.pack_root,
                dataset_root_id,
            )

        return self._readers[dataset_root_id]

    def read_rgb(self, row):
        data = self._reader(
            row["dataset_root_id"]
        ).read_bytes(row["relative_path"])

        with Image.open(io.BytesIO(data)) as image:
            return image.convert("RGB").copy()

    def close(self):
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
