"""
PyTorch training dataset backed by REIDPACK3.

Returns:
    images  : float32 [3, H, W], range [0, 1]
    targets : original Global integer class ID
    camids  : original camera/view metadata (string)
    domains : dataset root ID
    sample_uid: immutable sample identifier

This is the verified packed-data input layer.
FastReID Trainer integration is a separate step.

Important:
This module does not define training augmentations.
"""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import torch
from torch.utils.data import Dataset

from reidfm.data.v3_global_dataset import V3GlobalDataset


class V3PackedTrainDataset(Dataset):

    def __init__(
        self,
        manifest_path,
        pack_root,
        image_size=(256, 128),
    ):
        self.manifest_path = Path(manifest_path)
        self.pack_root = Path(pack_root)

        self.height, self.width = map(int, image_size)

        if self.height <= 0 or self.width <= 0:
            raise ValueError("Invalid image dimensions")

        self.records = []

        with self.manifest_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            reader = csv.DictReader(stream)

            required = {
                "sample_uid",
                "dataset_root_id",
                "relative_path",
                "class_id",
                "identity_key",
                "raw_camera_or_view",
            }

            if not required.issubset(
                set(reader.fieldnames or [])
            ):
                raise ValueError(
                    "Training manifest schema mismatch"
                )

            for row in reader:
                class_id = int(row["class_id"])

                if class_id < 0:
                    raise ValueError("Negative class ID")

                if not row["identity_key"]:
                    raise ValueError("Missing identity key")

                self.records.append({
                    "sample_uid": row["sample_uid"],
                    "dataset_root_id": row["dataset_root_id"],
                    "relative_path": row["relative_path"],
                    "identity_key": row["identity_key"],
                    "class_id": class_id,
                    "camera": row["raw_camera_or_view"],
                })

        if not self.records:
            raise ValueError("Empty training manifest")

        # Lazy-open inside the DataLoader worker.
        self._reader = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]

        if self._reader is None:
            self._reader = V3GlobalDataset(
                self.manifest_path,
                self.pack_root,
                "train",
            )

        image = self._reader.read_rgb(row)

        # Deterministic resize for integration test.
        # Training augmentations will be added by FastReID.
        from PIL import Image

        image = image.resize(
            (self.width, self.height),
            resample=Image.Resampling.BILINEAR,
        )

        arr = np.asarray(image, dtype=np.uint8)
        arr = np.ascontiguousarray(
            arr.transpose(2, 0, 1)
        )

        tensor = torch.from_numpy(
            arr.copy()
        ).float().div_(255.0)

        return {
            "images": tensor,
            "targets": row["class_id"],
            "camids": row["camera"],
            "domains": row["dataset_root_id"],
            "sample_uid": row["sample_uid"],
        }

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Do not send open file/SQLite handles to workers.
        state["_reader"] = None
        return state
