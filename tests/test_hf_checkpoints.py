"""HF checkpoint store condition allowlist."""

from __future__ import annotations

import pytest

from reidfm.experiments.contract import BACKBONE_STUDY_CONDITIONS, LOSS_STUDY_CONDITIONS
from scripts.reid_hf_checkpoints import ALLOWED_CHECKPOINT_CONDITIONS, HuggingFaceCheckpointStore


def test_allowed_conditions_include_loss_and_backbone_study():
    assert set(LOSS_STUDY_CONDITIONS) <= ALLOWED_CHECKPOINT_CONDITIONS
    assert set(BACKBONE_STUDY_CONDITIONS) <= ALLOWED_CHECKPOINT_CONDITIONS


def test_backbone_condition_accepted(tmp_path):
    HuggingFaceCheckpointStore(
        repo_id="reid-research-hub/reid-loss-study-live",
        token="hf_test_token",
        condition="ConvNeXtS_R0",
        seed=0,
        research_commit="abc",
        fingerprint="def",
        output=tmp_path,
        max_epoch=50,
        api=object(),
    )


def test_unknown_condition_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown study condition"):
        HuggingFaceCheckpointStore(
            repo_id="reid-research-hub/reid-loss-study-live",
            token="hf_test_token",
            condition="NotARealCondition",
            seed=0,
            research_commit="abc",
            fingerprint="def",
            output=tmp_path,
            api=object(),
        )
