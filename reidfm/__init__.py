"""reidfm: multi-source (foundation) person ReID fine-tuning on top of FastReID.

Light imports only; torch/FastReID are loaded lazily by the modules that need them.
"""

from reidfm.utils.paths import FASTREID_COMMIT, FASTREID_ROOT, REPO_ROOT

__version__ = "0.1.0"


def ensure_fastreid() -> None:
    from reidfm.fastreid_compat import ensure_fastreid as _ensure

    _ensure()


def available_models(**filters):
    from reidfm.models.registry import available_models as _available

    return _available(**filters)


def get_model_spec(key: str):
    from reidfm.models.registry import get_model_spec as _get

    return _get(key)


__all__ = [
    "FASTREID_COMMIT",
    "FASTREID_ROOT",
    "REPO_ROOT",
    "__version__",
    "available_models",
    "ensure_fastreid",
    "get_model_spec",
]
