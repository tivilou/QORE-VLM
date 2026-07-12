"""QORE KV-Cache: Quantum-optimized cache eviction for efficient LLM inference.

Compatibility note:
    These cache classes subclass transformers.DynamicCache and rely on its
    `key_cache` / `value_cache` list attributes. Transformers >= 4.54 refactored
    DynamicCache to use a `layers` structure and removed these attributes,
    which breaks the eviction logic. Pin transformers < 4.54 (see requirements.txt).
"""

import warnings


def _check_transformers_compat():
    """Warn early if the installed transformers version is incompatible."""
    try:
        import transformers
        from transformers.cache_utils import DynamicCache
        # The cache classes need the key_cache/value_cache list API
        probe = DynamicCache()
        if not hasattr(probe, "key_cache"):
            warnings.warn(
                f"transformers {transformers.__version__} removed "
                f"DynamicCache.key_cache (refactored in 4.54+). The QORE KV-Cache "
                f"classes require transformers < 4.54. Please run: "
                f"pip install 'transformers>=4.40,<4.54'  (see requirements.txt).",
                RuntimeWarning,
            )
    except Exception:
        pass


_check_transformers_compat()

from .qore_cache import QORECache
from .baselines.h2o_cache import H2OCache
from .baselines.snapkv_cache import SnapKVCache
from .baselines.pyramidkv_cache import PyramidKVCache
from .baselines.window_cache import WindowCache
from .baselines.random_cache import RandomCache
from .attention_capture import AttentionCapture

__all__ = [
    "QORECache", "H2OCache", "SnapKVCache", "PyramidKVCache",
    "WindowCache", "RandomCache", "AttentionCapture",
]
