"""YAML configuration helpers for enhancer pipelines."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .pipeline import EnhancerPipeline
from .registry import create_pipeline


def load_enhancer_config(
    source: str | Path | Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Load ``selection.enhancers`` from YAML or an existing mapping."""
    if isinstance(source, Mapping):
        document = dict(source)
    else:
        path = Path(source)
        with path.open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}

    selection = document.get("selection", document)
    if not isinstance(selection, Mapping):
        raise ValueError("selection must be a mapping")

    specs = selection.get("enhancers", selection.get("plugins"))
    if not isinstance(specs, list) or not specs:
        raise ValueError("configuration requires a non-empty selection.enhancers list")

    names: list[str] = []
    configs: dict[str, dict[str, Any]] = {}
    for index, spec in enumerate(specs):
        if isinstance(spec, str):
            name = spec
            config: dict[str, Any] = {}
        elif isinstance(spec, Mapping):
            name = spec.get("name")
            config = spec.get("config", {})
            if not isinstance(name, str) or not name:
                raise ValueError(f"enhancer at index {index} requires a name")
            if not isinstance(config, Mapping):
                raise ValueError(f"config for enhancer '{name}' must be a mapping")
            config = dict(config)
        else:
            raise ValueError(
                f"enhancer at index {index} must be a name or mapping"
            )

        if name in configs:
            raise ValueError(f"duplicate enhancer name in configuration: '{name}'")
        names.append(name)
        configs[name] = config

    return names, configs


def create_pipeline_from_config(
    source: str | Path | Mapping[str, Any],
) -> EnhancerPipeline:
    """Create a strictly validated pipeline from an experiment config."""
    names, configs = load_enhancer_config(source)
    return create_pipeline(names, configs, strict_composition=True)
