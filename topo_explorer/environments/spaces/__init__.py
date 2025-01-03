"""Manifold spaces registry and factory."""

from typing import Dict, Optional, Type
from .base_manifold import BaseManifold
from .sphere import SphereManifold
from .torus import TorusManifold
from .hyperbolic import HyperbolicManifold
from .klein import KleinBottleManifold
from .mobius import MobiusManifold
from .projective import ProjectiveManifold
from .n_torus import NTorusManifold

MANIFOLD_REGISTRY: Dict[str, Type[BaseManifold]] = {
    'sphere': SphereManifold,
    'torus': TorusManifold,
    'hyperbolic': HyperbolicManifold,
    'klein': KleinBottleManifold,
    'mobius': MobiusManifold,
    'projective': ProjectiveManifold,
    'n_torus': NTorusManifold,
}

def get_manifold(manifold_type: str, params: Optional[Dict] = None) -> BaseManifold:
    """
    Factory function to create manifold instances.
    
    Args:
        manifold_type: String identifier for the manifold type
        params: Optional parameters for manifold initialization
    
    Returns:
        Instance of specified manifold
        
    Raises:
        ValueError: If manifold_type is not recognized
    """
    if manifold_type not in MANIFOLD_REGISTRY:
        raise ValueError(
            f"Unknown manifold type: {manifold_type}. "
            f"Available types are: {list(MANIFOLD_REGISTRY.keys())}"
        )
    
    return MANIFOLD_REGISTRY[manifold_type](params)

__all__ = [
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