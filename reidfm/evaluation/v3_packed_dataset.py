"""Packed V3 validation Dataset with original FastReID preprocessing."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset

from reidfm.data.v3_pack_reader import V3PackReader


class V3PackedEvalDataset(Dataset):

    REQUIRED_FIELDS = (
        "sample_uid",
        "dataset",
        "dataset_root_id",
        "relative_path",
        "identity_key",
        "validation_protocol_id",
        "retrieval_role",
    )

    def __init__(self, records, pack_root, size_hw=(384, 128)):

        self.records = [dict(row) for row in records]

        if not self.records:
            raise ValueError("Empty V3 evaluation dataset")

        self.pack_root = Path(pack_root)

        if not self.pack_root.is_dir():
            raise FileNotFoundError(self.pack_root)

        self.size_hw = tuple(int(x) for x in size_hw)

        if len(self.size_hw) != 2 or min(self.size_hw) <= 0:
            raise ValueError("Invalid evaluation image dimensions")

        for index, row in enumerate(self.records):

            missing = [
                key for key in self.REQUIRED_FIELDS
                if not row.get(key)
            ]

            if missing:
                raise ValueError(
                    f"Record {index}: missing fields {missing}"
                )

            if row["retrieval_role"] not in ("query", "gallery"):
                raise ValueError(
                    f"Record {index}: invalid retrieval role"
                )

        self._readers = {}
        self._owner_pid = os.getpid()

    def __len__(self):
        return len(self.records)

    def _get_reader(self, domain):

        current_pid = os.getpid()

        if current_pid != self._owner_pid:

            # Never use inherited SQLite connections or file handles.
            # A dataset with opened readers must not be forked.
            if self._readers:
                raise RuntimeError(
                    "Evaluation Dataset was forked with open PACK readers; "
                    "create a fresh Dataset before starting workers"
                )

            self._owner_pid = current_pid

        if domain not in self._readers:

            self._readers[domain] = V3PackReader(
                self.pack_root,
                domain,
            )

        return self._readers[domain]

    def __getitem__(self, index):

        if not 0 <= index < len(self.records):
            raise IndexError(index)

        row = self.records[index]

        raw = self._get_reader(
            row["dataset_root_id"]
        ).read_bytes(
            row["relative_path"]
        )

        image = cv2.imdecode(
            np.frombuffer(raw, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                f"Cannot decode image: {row['sample_uid']}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        height, width = self.size_hw

        image = cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        ).astype(np.float32)

        tensor = torch.from_numpy(
            np.ascontiguousarray(
                image.transpose(2, 0, 1)
            )
        )

        return {
            "images": tensor,
            "index": index,
            "sample_uid": row["sample_uid"],
            "dataset": row["dataset"],
            "dataset_root_id": row["dataset_root_id"],
            "identity_key": row["identity_key"],
            "validation_protocol_id": row["validation_protocol_id"],
            "retrieval_role": row["retrieval_role"],
            "raw_camera_or_view": row.get("raw_camera_or_view", ""),
            "camera_context_id": row.get("camera_context_id", ""),
            "tracklet_id": row.get("tracklet_id", ""),
        }

    def close(self):

        if os.getpid() != self._owner_pid:
            self._readers = {}
            return

        for reader in self._readers.values():
            reader.close()

        self._readers.clear()

    def __getstate__(self):

        state = self.__dict__.copy()

        # Spawned workers must create their own PACK readers.
        state["_readers"] = {}
        state["_owner_pid"] = None

        return state
