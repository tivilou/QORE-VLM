"""Registry for QUBO enhancers: discovery, validation, and instantiation."""

import importlib
import pkgutil
from typing import Any

from .base import QUBOEnhancer
from .pipeline import EnhancerPipeline

# Global registry: name -> enhancer class
_REGISTRY: dict[str, type[QUBOEnhancer]] = {}
_DISCOVERED_PACKAGES: set[str] = set()


def discover_enhancers(package_name: str = "qore.enhancers") -> None:
    """Import plugin modules from a package without editing its ``__init__``."""
    if package_name in _DISCOVERED_PACKAGES:
        return

    package = importlib.import_module(package_name)
    skip = {"base", "config", "pipeline", "registry"}
    for module in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        leaf_name = module.name.rsplit(".", 1)[-1]
        if leaf_name.startswith("_") or leaf_name in skip:
            continue
        importlib.import_module(module.name)
    _DISCOVERED_PACKAGES.add(package_name)


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
    discover_enhancers()
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
    discover_enhancers()
    return sorted(_REGISTRY.keys())


def create_pipeline(
    enhancer_names: list[str],
    configs: dict[str, dict[str, Any]] | None = None,
    *,
    strict_composition: bool = False,
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
            ["baseline", "idea4"],
            configs={"baseline": {"gamma": 0.5}, "idea4": {"alpha": 0.1}}
        )
    """
    if not enhancer_names:
        raise ValueError("Pipeline requires at least one enhancer name")

    discover_enhancers()
    configs = configs or {}
    enhancers = []

    for name in enhancer_names:
        config = configs.get(name, {})
        enhancers.append(get_enhancer(name, config))

    return EnhancerPipeline(
        enhancers,
        strict_composition=strict_composition,
    )
