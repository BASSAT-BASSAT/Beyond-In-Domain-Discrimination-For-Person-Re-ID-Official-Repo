"""Crash-aware Hugging Face checkpoint transport for the ReID V3 study.

This transports the original FastReID checkpoints unchanged. It does not
implement model, sampler, AMP, or DDP restoration itself: that is the job of
the pinned research trainer after this module installs ``last_checkpoint``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from reidfm.experiments.contract import BACKBONE_STUDY_CONDITIONS, LOSS_STUDY_CONDITIONS


SCHEMA = 1
VALID_NAMES = {"model_mid.pth", "model_last.pth", "model_final.pth"}

ALLOWED_CHECKPOINT_CONDITIONS = frozenset((*LOSS_STUDY_CONDITIONS, *BACKBONE_STUDY_CONDITIONS))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_copy(source: Path, destination: Path, marker: Path | None = None) -> None:
    """Copy only a complete, unchanged file; never publish a growing .pth."""
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    before = source.stat()
    if before.st_size < 1024:
        raise RuntimeError(f"Checkpoint too small: {source.name}")
    if marker and (time.time_ns() - before.st_mtime_ns) < 3_000_000_000:
        raise RuntimeError("Checkpoint is still settling; retry in a few seconds")
    marker_before = (marker.stat().st_mtime_ns, marker.read_text()) if marker else None
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".copying")
    try:
        with source.open("rb") as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        after = source.stat()
        marker_after = (marker.stat().st_mtime_ns, marker.read_text()) if marker else None
        if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                or marker_before != marker_after):
            raise RuntimeError("Checkpoint changed during snapshot; retry later")
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)


def inspect_checkpoint(path: Path, *, iters_per_epoch: int, max_epoch: int,
                       expected_run: str, expected_fingerprint: str) -> dict:
    """Read trusted local checkpoint metadata and enforce resumable state."""
    import torch

    # The bytes originate from our pinned private research trainer, or from
    # an explicitly permitted private team repository verified by SHA-256.
    # Restrict deserialization to tensor/primitive weight data.
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Expected FastReID checkpoint dictionary")
    for name in ("model", "optimizer", "scheduler", "v3_freeze_counter"):
        if name not in state:
            raise ValueError(f"Checkpoint is not fully resumable: missing {name}")
    if path.name == "model_mid.pth":
        from reidfm.engine.v3_mid_epoch_checkpoint import resolve_v3_resume_cursor

        if "v3_multirank_bundle" not in state:
            raise ValueError("Midpoint checkpoint missing two-rank resume bundle")
        bundle = state["v3_multirank_bundle"]
        if (bundle.get("run_id") != expected_run
                or bundle.get("fingerprint") != expected_fingerprint
                or bundle.get("world_size") != 2):
            raise ValueError("Midpoint is from another run, code, or GPU configuration")
        epoch, within, completed = resolve_v3_resume_cursor(
            state, iters_per_epoch, max_epoch
        )
    else:
        # FastReID's epoch checkpoints resume at the next epoch boundary.
        if "v3_resume_schema" in state or "v3_multirank_bundle" in state:
            raise ValueError("Epoch checkpoint contains unexpected midpoint cursor")
        epoch_index = state.get("epoch")
        if type(epoch_index) is not int or not 0 <= epoch_index < max_epoch:
            raise ValueError("Invalid end-of-epoch checkpoint cursor")
        epoch, within = epoch_index + 1, 0
        completed = epoch * iters_per_epoch
    result = {"epoch": epoch, "iteration_in_epoch": within,
              "global_steps": completed}
    del state
    return result


class HuggingFaceCheckpointStore:
    def __init__(self, *, repo_id: str, token: str, condition: str, seed: int,
                 research_commit: str, fingerprint: str, output: Path,
                 iters_per_epoch: int = 2000, max_epoch: int = 60,
                 session_tag: str | None = None, api=None):
        if not token:
            raise ValueError("HF_TOKEN is empty. Enable it in Kaggle Secrets.")
        if condition not in ALLOWED_CHECKPOINT_CONDITIONS:
            raise ValueError(
                f"Unknown study condition {condition!r}; "
                f"expected one of {sorted(ALLOWED_CHECKPOINT_CONDITIONS)}"
            )
        if seed not in (0, 1, 2):
            raise ValueError("Unknown seed")
        if not repo_id.startswith("reid-research-hub/"):
            raise ValueError("Expected checkpoint repository in the approved organization")
        self.repo_id = repo_id
        self.token = token
        self.condition = condition
        self.seed = seed
        self.research_commit = research_commit
        self.fingerprint = fingerprint
        self.output = Path(output)
        self.prefix = f"{condition}/seed{seed}"
        self.iters_per_epoch = int(iters_per_epoch)
        self.max_epoch = int(max_epoch)
        self.latest = None
        self.best_signature = None
        self.metrics_signature = None
        self.session_tag = session_tag or str(int(time.time()))
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
        self.api = api

    def _key(self, item: str) -> str:
        return f"{self.prefix}/{item}"

    def ensure_private_repository(self) -> None:
        from huggingface_hub.errors import RepositoryNotFoundError

        try:
            info = self.api.repo_info(repo_id=self.repo_id, repo_type="model")
        except RepositoryNotFoundError:
            # Only the first authorized team member needs to create it.
            try:
                self.api.create_repo(
                    repo_id=self.repo_id, repo_type="model", private=True, exist_ok=True
                )
                info = self.api.repo_info(repo_id=self.repo_id, repo_type="model")
            except Exception as exc:
                raise RuntimeError(
                    "Cannot access/create the private HF checkpoint repository. "
                    "An organization admin must create it once and grant write access."
                ) from exc
        if not bool(getattr(info, "private", False)):
            raise RuntimeError("Refusing to push research checkpoints to a public HF repo")

    def _read_json(self, key: str) -> dict | None:
        if not self.api.file_exists(self.repo_id, key, repo_type="model"):
            return None
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(
            repo_id=self.repo_id, filename=key, repo_type="model", token=self.token
        )
        doc = json.loads(Path(local).read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise ValueError("Malformed HF checkpoint metadata")
        return doc

    def _validate_identity(self, doc: dict) -> None:
        expected = {
            "schema": SCHEMA, "condition": self.condition, "seed": self.seed,
            "research_commit": self.research_commit,
            "config_fingerprint": self.fingerprint,
        }
        for key, value in expected.items():
            if doc.get(key) != value:
                raise RuntimeError(f"HF checkpoint identity mismatch: {key}")
        if doc.get("filename") not in VALID_NAMES:
            raise ValueError("Unsafe checkpoint filename in HF metadata")
        if doc.get("remote_file") not in {
            self._key("slot_a.pth"), self._key("slot_b.pth")
        }:
            raise ValueError("Unexpected HF checkpoint path")
        if type(doc.get("global_steps")) is not int or not (0 < doc["global_steps"] <= self.max_epoch * self.iters_per_epoch):
            raise ValueError("Invalid remote training cursor")

    def restore_latest(self) -> dict | None:
        """Missing manifest means fresh; *any* other failure aborts the run."""
        self.output.mkdir(parents=True, exist_ok=True)
        doc = self._read_json(self._key("latest.json"))
        if doc is not None:
            self._validate_identity(doc)
        local_pointer = self.output / "last_checkpoint"
        if local_pointer.is_file():
            local_name = local_pointer.read_text(encoding="utf-8").strip()
            if local_name not in VALID_NAMES:
                raise RuntimeError("Unexpected local last_checkpoint pointer; refusing fresh start")
            if not (self.output / local_name).is_file():
                raise RuntimeError("Local last_checkpoint points to a missing file")
            local_cursor = inspect_checkpoint(
                self.output / local_name,
                iters_per_epoch=self.iters_per_epoch, max_epoch=self.max_epoch,
                expected_run=f"{self.condition}/seed_{self.seed}",
                expected_fingerprint=self.fingerprint,
            )
            if doc is None or local_cursor["global_steps"] >= doc["global_steps"]:
                self.latest = doc
                return {**(doc or {}), **local_cursor, "source": "local"}
        if doc is None:
            return None
        from huggingface_hub import hf_hub_download

        download = Path(hf_hub_download(
            repo_id=self.repo_id, filename=doc["remote_file"],
            repo_type="model", token=self.token
        ))
        if download.stat().st_size != doc["size"] or digest(download) != doc["sha256"]:
            raise RuntimeError("Remote checkpoint checksum failed. Refusing fresh start.")
        target = self.output / doc["filename"]
        with tempfile.TemporaryDirectory(dir=self.output) as tmp:
            staged = Path(tmp) / "restored.pth"
            safe_copy(download, staged)
            # Verify that the bytes pointed to by latest.json have the same
            # training cursor, full optimizer state, and V3 identity.
            check = staged.with_name(doc["filename"])
            os.replace(staged, check)
            actual_cursor = inspect_checkpoint(
                check, iters_per_epoch=self.iters_per_epoch, max_epoch=self.max_epoch,
                expected_run=f"{self.condition}/seed_{self.seed}",
                expected_fingerprint=self.fingerprint,
            )
            if actual_cursor["global_steps"] != doc["global_steps"]:
                raise RuntimeError("HF metadata cursor differs from checkpoint bytes")
            os.replace(check, target)
        # The trainer handles all original state, including the V3 midpoint bundle.
        (self.output / "last_checkpoint").write_text(target.name, encoding="utf-8")
        self.latest = doc
        return doc

    def _base_metadata(self) -> dict:
        return {
            "schema": SCHEMA, "condition": self.condition, "seed": self.seed,
            "research_commit": self.research_commit,
            "config_fingerprint": self.fingerprint,
        }

    def publish_latest(self) -> dict | None:
        """Two slots + last JSON pointer: interrupted upload cannot replace live state."""
        pointer = self.output / "last_checkpoint"
        if not pointer.is_file():
            return None
        filename = pointer.read_text(encoding="utf-8").strip()
        if filename not in VALID_NAMES:
            # model_best may briefly be FastReID's pointer before model_last.
            return None
        source = self.output / filename
        if not source.is_file():
            return None
        with tempfile.TemporaryDirectory(dir=self.output) as temp:
            staged = Path(temp) / filename
            safe_copy(source, staged, marker=pointer)
            cursor = inspect_checkpoint(
                staged, iters_per_epoch=self.iters_per_epoch,
                max_epoch=self.max_epoch,
                expected_run=f"{self.condition}/seed_{self.seed}",
                expected_fingerprint=self.fingerprint,
            )
            if self.latest and cursor["global_steps"] <= self.latest["global_steps"]:
                return None
            slot = "slot_b.pth" if self.latest and self.latest["remote_file"].endswith("slot_a.pth") else "slot_a.pth"
            remote_file = self._key(slot)
            doc = {
                **self._base_metadata(), **cursor, "filename": filename,
                "remote_file": remote_file, "size": staged.stat().st_size,
                "sha256": digest(staged), "saved_at_unix": int(time.time()),
            }
            # Upload bytes first. Publish the small authoritative pointer LAST.
            self.api.upload_file(
                path_or_fileobj=str(staged), path_in_repo=remote_file,
                repo_id=self.repo_id, repo_type="model",
                commit_message=f"{self.condition} seed{self.seed}: snapshot at step {cursor['global_steps']}",
            )
            self.api.upload_file(
                path_or_fileobj=json.dumps(doc, sort_keys=True).encode("utf-8"),
                path_in_repo=self._key("latest.json"), repo_id=self.repo_id,
                repo_type="model", commit_message=f"Publish step {cursor['global_steps']} as latest",
            )
            self.latest = doc
            return doc

    def publish_best(self) -> bool:
        """Keep the research trainer's best-validation model on HF too."""
        source = self.output / "model_best.pth"
        if not source.is_file():
            return False
        signature = (source.stat().st_size, source.stat().st_mtime_ns)
        if signature == self.best_signature:
            return False
        with tempfile.TemporaryDirectory(dir=self.output) as temp:
            staged = Path(temp) / source.name
            safe_copy(source, staged)
            if signature != (source.stat().st_size, source.stat().st_mtime_ns):
                raise RuntimeError("Best checkpoint changed during snapshot")
            best_meta = {
                **self._base_metadata(), "sha256": digest(staged),
                "size": staged.stat().st_size, "filename": source.name,
                "saved_at_unix": int(time.time()),
            }
            import torch
            best_state = torch.load(str(staged), map_location="cpu", weights_only=True)
            best_meta["best_epoch_index"] = int(best_state.get("epoch", -1))
            best_meta["validation_metric"] = float(best_state.get("metric", -1.0))
            del best_state
            self.api.upload_file(
                path_or_fileobj=str(staged), path_in_repo=self._key("model_best.pth"),
                repo_id=self.repo_id, repo_type="model",
                commit_message=f"Update {self.condition} seed{self.seed} best model",
            )
            self.api.upload_file(
                path_or_fileobj=json.dumps(best_meta, sort_keys=True).encode("utf-8"),
                path_in_repo=self._key("best.json"), repo_id=self.repo_id,
                repo_type="model", commit_message="Publish best-model manifest",
            )
        self.best_signature = signature
        return True

    def publish_metrics(self, force: bool = False) -> bool:
        """Preserve JSONL validation/training history under this session's ID."""
        source = self.output / "metrics.json"
        if not source.is_file():
            return False
        signature = (source.stat().st_size, source.stat().st_mtime_ns)
        if not force and signature == self.metrics_signature:
            return False
        with tempfile.TemporaryDirectory(dir=self.output) as temp:
            staged = Path(temp) / "metrics.json"
            safe_copy(source, staged)
            remote = self._key(f"sessions/{self.session_tag}/metrics.json")
            self.api.upload_file(
                path_or_fileobj=str(staged), path_in_repo=remote,
                repo_id=self.repo_id, repo_type="model",
                commit_message=f"Save {self.condition} seed{self.seed} session metrics",
            )
        self.metrics_signature = signature
        return True
