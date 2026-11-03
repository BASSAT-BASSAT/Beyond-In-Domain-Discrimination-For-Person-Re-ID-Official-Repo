"""V3 packed evaluation feature extraction."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from reidfm.evaluation.v3_packed_dataset import V3PackedEvalDataset


def extract_v3_packed_features(
    model,
    records,
    pack_root,
    *,
    size_hw=(384, 128),
    batch_size=4,
    num_workers=0,
    device="cpu",
    reidraw_files=None,
):
    if batch_size < 1 or num_workers < 0:
        raise ValueError("Invalid DataLoader settings")

    dataset = V3PackedEvalDataset(
        records,
        pack_root,
        size_hw=size_hw,
        reidraw_files=reidraw_files,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    original_training = bool(model.training)
    features = []
    seen = 0
    feature_dim = None

    try:
        model.eval()

        with torch.inference_mode():

            for batch in loader:
                indices = batch["index"].tolist()

                expected_indices = list(
                    range(seen, seen + len(indices))
                )

                if indices != expected_indices:
                    raise RuntimeError(
                        "Manifest order changed during extraction"
                    )

                expected_uids = [
                    dataset.records[i]["sample_uid"]
                    for i in indices
                ]

                if list(batch["sample_uid"]) != expected_uids:
                    raise RuntimeError(
                        "Sample UID alignment failed"
                    )

                images = batch["images"].to(device)

                output = model({"images": images})

                if (
                    not isinstance(output, torch.Tensor)
                    or output.ndim != 2
                    or output.shape[0] != len(indices)
                ):
                    raise ValueError(
                        "Invalid model feature output"
                    )

                if feature_dim is None:
                    feature_dim = output.shape[1]

                if (
                    feature_dim < 1
                    or output.shape[1] != feature_dim
                    or not torch.isfinite(output).all().item()
                ):
                    raise ValueError(
                        "Non-finite or inconsistent features"
                    )

                features.append(
                    output.float().cpu().numpy()
                )

                seen += len(indices)

    finally:
        model.train(original_training)
        dataset.close()

    if seen != len(dataset):
        raise RuntimeError(
            "Incomplete extraction coverage"
        )

    array = np.concatenate(
        features,
        axis=0,
    ).astype(np.float32, copy=False)

    return array, [
        dict(row) for row in dataset.records
    ]
