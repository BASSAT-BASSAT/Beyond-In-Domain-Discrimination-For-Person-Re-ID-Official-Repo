"""Model registry, zoo entries, init-weight conversion, and extra FastReID backbones."""

from reidfm.models.registry import MODEL_REGISTRY, ModelSpec, available_models, get_model_spec, iter_specs, register_model

__all__ = ["MODEL_REGISTRY", "ModelSpec", "available_models", "get_model_spec", "iter_specs", "register_model"]
