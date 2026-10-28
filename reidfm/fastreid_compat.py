"""Make the pinned FastReID clone importable and register reidfm's extensions into it.

Every entry point that touches FastReID calls :func:`ensure_fastreid` first. It:

1. puts ``./fast-reid`` on ``sys.path`` (FastReID is cloned, not pip-installed),
2. patches ``Checkpointer._load_file`` so torch >= 2.6 (``weights_only=True`` by default)
   can still load full FastReID checkpoints (optimizer/scheduler state included),
3. registers the manifest datasets, the domain-balanced sampler, and extra backbones.
"""

from __future__ import annotations

import sys

from reidfm.utils.paths import FASTREID_ROOT


def _patch_legacy_collections() -> None:
    """Restore aliases expected by older FastReID imports.

    This modifies only the current Python process.
    The pinned FastReID source remains unchanged.
    """
    import collections
    import collections.abc

    for name in (
        "Mapping",
        "MutableMapping",
        "Sequence",
        "MutableSequence",
        "Set",
        "MutableSet",
        "Iterable",
        "Iterator",
        "Callable",
    ):
        if not hasattr(collections, name):
            setattr(
                collections,
                name,
                getattr(collections.abc, name),
            )

_READY = False


def _put_fastreid_on_path() -> None:
    try:
        import fastreid  # noqa: F401

        return
    except ImportError:
        pass
    if not (FASTREID_ROOT / "fastreid").is_dir():
        raise ImportError(
            f"FastReID not found at {FASTREID_ROOT}. Run `python scripts/setup_fastreid.py` "
            "(or set REIDFM_FASTREID_ROOT to an existing clone)."
        )
    sys.path.insert(0, str(FASTREID_ROOT))
    import fastreid  # noqa: F401


def _patch_checkpoint_loading() -> None:
    import torch
    from fastreid.utils.checkpoint import Checkpointer

    if getattr(Checkpointer._load_file, "_reidfm_patched", False):
        return

    def _load_file(self, f: str):
        return torch.load(f, map_location=torch.device("cpu"), weights_only=False)

    _load_file._reidfm_patched = True
    Checkpointer._load_file = _load_file


def ensure_fastreid() -> None:
    """Idempotent: import FastReID, apply the compat patch, register reidfm extensions."""
    global _READY
    if _READY:
        return
    _patch_legacy_collections()
    _put_fastreid_on_path()
    _patch_checkpoint_loading()

    from reidfm.data.fastreid_datasets import register_manifest_datasets
    from reidfm.data.samplers import register_samplers
    from reidfm.models.backbones import register_backbones

    register_manifest_datasets()
    register_samplers()
    register_backbones()
    _register_reidfm_modeling()
    _READY = True


def _register_reidfm_modeling() -> None:
    from reidfm.modeling.heads.angular_head import AngularEmbeddingHead  # noqa: F401
    from reidfm.modeling.meta_arch.loss_study_baseline import LossStudyBaseline  # noqa: F401
