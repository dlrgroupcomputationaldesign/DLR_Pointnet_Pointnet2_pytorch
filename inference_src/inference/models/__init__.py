from importlib import import_module

__all__ = ["get_model_module", "AVAILABLE_MODELS"]

AVAILABLE_MODELS = {
    "pointnet2_sem_seg": ".pointnet2_sem_seg",
    "pointnet_sem_seg": ".pointnet_sem_seg",
}

def get_model_module(name: str):
    """
    Returns the imported module object for a given model name.
    Usage: mod = get_model_module("pointnet2_sem_seg")
    """
    if name in AVAILABLE_MODELS:
        return import_module(AVAILABLE_MODELS[name], package=__name__)

    # Also allow passing a full dotted path directly
    # e.g. "inference.models.pointnet2_sem_seg"
    if "." in name:
        return import_module(name)

    raise ValueError(
        f"Unknown model '{name}'. Available: {sorted(AVAILABLE_MODELS.keys())}"
    )
