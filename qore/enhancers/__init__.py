"""QUBO enhancer plugin framework."""

from .base import QUBOEnhancer
from .registry import (
    create_pipeline,
    discover_enhancers,
    get_enhancer,
    list_enhancers,
    register_enhancer,
)
from .pipeline import EnhancerPipeline
from .config import create_pipeline_from_config, load_enhancer_config

# Discover modules in this package. Adding a plugin no longer requires editing
# this file; importing the package is enough to register it.
discover_enhancers()

__all__ = [
    "QUBOEnhancer",
    "EnhancerPipeline",
    "register_enhancer",
    "discover_enhancers",
    "get_enhancer",
    "list_enhancers",
    "create_pipeline",
    "load_enhancer_config",
    "create_pipeline_from_config",
]
