"""Registry for QUBO enhancers: dynamic discovery and instantiation."""

from typing import Any

from .base import QUBOEnhancer
from .pipeline import EnhancerPipeline

# Global registry: name -> enhancer class
_REGISTRY: dict[str, type[QUBOEnhancer]] = {}


def register_enhancer(name: str):
    """
    Decorator to register an enhancer class.

    Usage:
        @register_enhancer("my_idea")
        class MyIdeaEnhancer(QUBOEnhancer):
            ...

    Args:
        name: Unique identifier for this enhancer.

    Returns:
        Decorator function that registers the class.
    """
    def decorator(cls: type[QUBOEnhancer]) -> type[QUBOEnhancer]:
        if name in _REGISTRY:
            raise ValueError(f"Enhancer '{name}' is already registered")
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_enhancer(name: str, config: dict[str, Any] | None = None) -> QUBOEnhancer:
    """
    Get an enhancer instance by name.

    Args:
        name: Enhancer name (must be registered).
        config: Configuration dict to pass to the enhancer constructor.

    Returns:
        Instantiated QUBOEnhancer.

    Raises:
        ValueError: If the enhancer name is not registered.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown enhancer: '{name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](config)


def list_enhancers() -> list[str]:
    """
    List all registered enhancer names.

    Returns:
        List of enhancer names that can be used with get_enhancer().
    """
    return list(_REGISTRY.keys())


def create_pipeline(
    enhancer_names: list[str],
    configs: dict[str, dict[str, Any]] | None = None,
) -> EnhancerPipeline:
    """
    Create an enhancer pipeline from a list of names.

    Args:
        enhancer_names: List of enhancer names in order.
        configs: Optional dict mapping enhancer names to their config dicts.

    Returns:
        EnhancerPipeline instance.

    Example:
        pipeline = create_pipeline(
            ["baseline", "idea6"],
            configs={"baseline": {"gamma": 0.5}, "idea6": {"delta": 0.1}}
        )
    """
    if not enhancer_names:
        raise ValueError("Pipeline requires at least one enhancer name")

    configs = configs or {}
    enhancers = []

    for name in enhancer_names:
        config = configs.get(name, {})
        enhancers.append(get_enhancer(name, config))

    return EnhancerPipeline(enhancers)
