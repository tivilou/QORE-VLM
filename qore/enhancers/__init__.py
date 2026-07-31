"""QUBO enhancers: pluggable ideas for modifying interaction matrices."""

from .base import QUBOEnhancer
from .registry import register_enhancer, get_enhancer, list_enhancers, create_pipeline
from .pipeline import EnhancerPipeline

# Import enhancers to trigger registration
from . import baseline
from . import idea6_complementarity
from . import idea4_context_integrity
from . import idea7_differentiable_qubo

__all__ = [
    "QUBOEnhancer",
    "EnhancerPipeline",
    "register_enhancer",
    "get_enhancer",
    "list_enhancers",
    "create_pipeline",
]
