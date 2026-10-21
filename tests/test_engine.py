"""Kaggle session helpers: checkpoint pruning, run-dir restore, time guard decisions."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_resolve_resume():
    from reidfm.engine.checkpointing import resolve_resume

    assert not resolve_resume(fresh=True, auto_resume=True, output_dir="/x", resume_flag=False)
    assert resolve_resume(fresh=False, auto_resume=False, output_dir="/x", resume_flag=True)
    assert not resolve_resume(fresh=False, auto_resume=True, output_dir="/x", resume_flag=False)


def test_resolve_resume_when_last_checkpoint_exists(tmp_path: Path):
    from reidfm.engine.checkpointing import resolve_resume

    _touch(tmp_path / "last_checkpoint", "model_0000.pth")
    assert resolve_resume(fresh=False, auto_resume=True, output_dir=tmp_path, resume_flag=False)
    assert not resolve_resume(fresh=True, auto_resume=True, output_dir=tmp_path, resume_flag=False)


@pytest.mark.fastreid
def test_prune_keeps_newest_and_protected(tmp_path: Path):
    from reidfm.engine.hooks import CheckpointPruneHook

    for epoch in range(5):
        _touch(tmp_path / f"model_{epoch:04d}.pth")
    _touch(tmp_path / "model_best.pth")
    _touch(tmp_path / "last_checkpoint", "model_0001.pth")
    CheckpointPruneHook(str(tmp_path), keep=2).after_epoch()
    left = sorted(p.name for p in tmp_path.glob("model_*.pth"))
    assert left == ["model_0001.pth", "model_0003.pth", "model_0004.pth", "model_best.pth"]


@pytest.mark.fastreid
def test_restore_run_dir_copies_what_resume_needs(tmp_path: Path):
    from reidfm.engine.resume import restore_run_dir

    src = tmp_path / "input" / "run"
    _touch(src / "model_0002.pth")
    _touch(src / "model_0001.pth")
    _touch(src / "model_best.pth")
    _touch(src / "last_checkpoint", "model_0002.pth")
    _touch(src / "config.yaml", "a: 1")
    dst = tmp_path / "working" / "run"
    restore_run_dir(src, dst)
    names = {p.name for p in dst.iterdir()}
    assert {"model_0002.pth", "model_best.pth", "last_checkpoint", "config.yaml"} <= names
    assert "model_0001.pth" not in names


@pytest.mark.fastreid
def test_time_guard_stops_when_next_epoch_does_not_fit(monkeypatch):
    from reidfm.engine import hooks

    saved = []
    checkpointer = SimpleNamespace(save=lambda name, **kw: saved.append((name, kw)))
    trainer = SimpleNamespace(epoch=3, max_epoch=30, saved_epoch=-1, stop_requested=False, _hooks=[])
    monkeypatch.setenv("REIDFM_SESSION_START", str(time.time() - 3500))
    guard = hooks.TimeGuardHook(budget_h=1.0, checkpointer=checkpointer)
    guard.trainer = trainer
    guard._epoch_start = time.time() - 200  # a 200 s epoch; only ~100 s left
    guard.after_epoch()
    assert trainer.stop_requested
    assert saved and saved[0][0] == "model_0003"

    trainer.stop_requested = False
    monkeypatch.setenv("REIDFM_SESSION_START", str(time.time()))
    guard._epoch_start = time.time() - 10
    guard.after_epoch()
    assert not trainer.stop_requested
