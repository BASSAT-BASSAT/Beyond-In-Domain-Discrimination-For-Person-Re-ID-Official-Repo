"""
FastReID-compatible training input for Global P3/P5.

Uses:
  - original global class IDs (no implicit relabel)
  - original V3PackReader
  - original FastReID transforms when supplied
  - explicit domain and camera metadata

No Pool V2 conversion or duplicate training loop.
"""

from __future__ import annotations

from pathlib import Path
import csv
import io
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from reidfm.data.v3_pack_reader import V3PackReader
from reidfm.data.samplers import v3_identity_key


class V3FastReIDTrainDataset(Dataset):

    def __init__(
        self,
        manifest_path,
        pack_root,
        transform=None,
        records=None,
        virtual_len=0,
        image_size=(256, 128),
    ):
        self.manifest_path = Path(manifest_path)
        self.pack_root = Path(pack_root)
        self.transform = transform
        self.virtual_len = int(virtual_len)
        self.image_size = tuple(image_size)

        if self.virtual_len < 0:
            raise ValueError("Negative virtual length")

        if records is None:
            with self.manifest_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as f:
                records = list(csv.DictReader(f))

        self.records = list(records)

        if not self.records:
            raise ValueError("Empty V3 training dataset")

        self._readers = {}
        self._owner_pid = os.getpid()

        # Immutable sampler input. Use global class ID
        # with an explicit domain-aware identity key.
        self.img_items = []

        for row in self.records:
            cid = int(row["class_id"])

            if not (0 <= cid < 10036):
                raise ValueError("Invalid Global class ID")

            domain = row["dataset_root_id"]

            pid = v3_identity_key(domain, cid)

            self.img_items.append((
                row["relative_path"],
                pid,
                str(row.get("raw_camera_or_view", "")),
            ))

        # Required by FastReID DefaultTrainer.auto_scale_hyperparams().
        # The V3 setup config validates the full Global class mapping.
        self.num_classes = len({int(row["class_id"]) for row in self.records})
    def __len__(self):
        return self.virtual_len or len(self.records)

    def _reader(self, domain):
        pid = os.getpid()

        if pid != self._owner_pid:
            # A worker must never reuse inherited
            # SQLite connections or PACK file handles.
            self._readers = {}
            self._owner_pid = pid

        if domain not in self._readers:
            self._readers[domain] = V3PackReader(
                self.pack_root, domain
            )

        return self._readers[domain]

    def __getitem__(self, index):
        # Virtual length is used by the Trainer for
        # epoch accounting, not for sampling indices.
        if not 0 <= index < len(self.records):
            raise IndexError(index)

        row = self.records[index]
        domain = row["dataset_root_id"]
        cid = int(row["class_id"])

        data = self._reader(domain).read_bytes(
            row["relative_path"]
        )

        with Image.open(io.BytesIO(data)) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)
            else:
                height, width = self.image_size
                image = image.resize(
                    (width, height),
                    Image.Resampling.BILINEAR,
                )
                array = np.ascontiguousarray(
                    np.asarray(image).transpose(2, 0, 1)
                )
                image = torch.from_numpy(array.copy())

        if not isinstance(image, torch.Tensor):
            raise TypeError(
                "Transform must return a Tensor"
            )

        camera_raw = str(
            row.get("raw_camera_or_view", "")
        )

        return {
            "images": image,
            "targets": cid,
            # Unknown camera is not fabricated as
            # a valid cross-camera identity.
            "camids": int(
                camera_raw
            ) if camera_raw.isdigit() else -1,
            "img_paths": (
                f"pack://{domain}/"
                f"{row['relative_path']}"
            ),
        }

    def close(self):
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_readers"] = {}
        state["_owner_pid"] = None
        return state
