# Plugin Architecture v2

QORE research ideas plug into passage selection without adding idea-specific
branches to `applications/rag/selector.py`. Retrieval, generation, evaluation,
and the K-budget solver remain shared controls.

## Composition contract

Every `QUBOEnhancer` declares one composition mode:

- `replace`: defines a complete root interaction objective.
- `add`: preserves the current objective and contributes another term.

A strict pipeline contains at most one `replace` plugin, and it must be first.
This prevents a later plugin from silently erasing earlier research terms.
Programmatic pipelines retain the legacy overwrite behavior with a deprecation
warning. YAML pipelines use strict validation.

## YAML experiments

```python
from qore.enhancers import create_pipeline_from_config

pipeline = create_pipeline_from_config(
    "configs/experiments/plugin_pipeline_v2.yaml"
)
w, trace = pipeline.enhance_with_diagnostics(a, b, context)
```

The loader accepts either `selection.enhancers` or top-level `enhancers`.
Duplicate names and ambiguous root objectives are rejected before an experiment
starts.

## Adding an idea

1. Add one module under `qore/enhancers/`.
2. Subclass `QUBOEnhancer` and register it with `@register_enhancer("name")`.
3. Return a symmetric, finite `(N, N)` matrix with a zero diagonal.
4. Declare `composition_mode` and `required_context_keys`.
5. Add contract, composition, and fixed-seed regression tests.

Plugin modules are discovered automatically. Do not edit
`qore/enhancers/__init__.py` when adding or removing an idea.

## Five-idea boundary

The cohesion and Mobius-QUBO ideas can contribute interaction terms through
enhancers. Adaptive VoI may contribute a quality transformation in the next
plugin phase. Saturating submodular and spectral/DPP selectors replace the
terminal selection strategy and must not be disguised as additive QUBO terms.
All variants must emit paired quality, redundancy, and latency diagnostics.

## Backward compatibility

Existing `select_passages` calls and legacy `gamma`, `delta`, and
`complementarity_method` arguments keep their previous numerical behavior.
New experiment configuration should use strict YAML pipelines.
