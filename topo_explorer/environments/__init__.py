"""Environments module initialization."""

from .manifold_env import ManifoldEnvironment
from .spaces import (
    BaseManifold,
    SphereManifold,
    TorusManifold,
    HyperbolicManifold,
    KleinBottleManifold,
    MobiusManifold,
    ProjectiveManifold,
    NTorusManifold,
    get_manifold,
    MANIFOLD_REGISTRY
)

__all__ = [
    "ManifoldEnvironment",
    "BaseManifold",
    "SphereManifold",
    "TorusManifold",
    "HyperbolicManifold",
    "KleinBottleManifold",
    "MobiusManifold",
    "ProjectiveManifold",
    "NTorusManifold",
    "get_manifold",
    "MANIFOLD_REGISTRY"
]