"""Checkpoint discovery and auto-resume helpers."""

from __future__ import annotations

from pathlib import Path


def run_has_checkpoint(output_dir: str | Path) -> bool:
    """True if ``output_dir`` contains a FastReID ``last_checkpoint`` pointer."""
    return (Path(output_dir) / "last_checkpoint").is_file()


def resolve_resume(fresh: bool, auto_resume: bool, output_dir: str | Path, resume_flag: bool) -> bool:
    """Whether to pass ``resume=True`` into the trainer.

    * ``--fresh`` → never resume.
    * explicit ``--resume`` → always resume (if a checkpoint exists; FastReID errors otherwise).
    * ``auto_resume`` and an existing run → resume without passing ``--resume``.
    """
    if fresh:
        return False
    if resume_flag:
        return True
    return auto_resume and run_has_checkpoint(output_dir)
