from __future__ import annotations

import copy

import pytest
import torch

from applications.rag.generator_native_evidence_anchor_transport import (
    BoundaryError,
    ResidualAnchorHook,
    ResidualPluginRegistry,
    assert_prompt_token_identity,
    build_compact_boundary_manifest,
    build_prompt_token_span_batch_with_fallback,
    build_prompt_token_span_map,
    find_forbidden_fields,
    validate_plugin_allowlist,
)


def _fixture():
    prompts = (
        "alpha beta gamma delta",
        "one two three four five",
        "bad input",
    )
    tokenized = {
        "input_ids": [
            [101, 11, 12, 13, 14, 102, 0, 0],
            [0, 0, 201, 21, 22, 23, 24, 25],
            [301, 31, 32, 302, 0, 0, 0, 0],
        ],
        "attention_mask": [
            [1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ],
        "special_tokens_mask": [
            [1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0],
        ],
        "offset_mapping": [
            [[0, 0], [0, 5], [6, 10], [11, 16], [17, 22], [0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0], [0, 3], [4, 7], [8, 13], [14, 18], [19, 23]],
            [[0, 0], [0, 3], [4, 9], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        ],
    }
    spans = (
        ((0, 10),),
        ((0, 3),),
        ((99, 100),),
    )
    return prompts, tokenized, spans


def _valid_two_row_fixture():
    prompts, tokenized, spans = _fixture()
    return prompts[:2], {key: value[:2] for key, value in tokenized.items()}, spans[:2]


def test_batch_mapping_handles_right_and_left_padding_without_overlap():
    prompts, tokenized, spans = _valid_two_row_fixture()
    mapping = build_prompt_token_span_map(tokenized, prompts, spans)

    assert mapping.batch_size == 2
    assert mapping.sequence_length == 8
    assert mapping.reader_tokens == ((1, 2), (3,))
    assert mapping.control_tokens == ((3, 4), (4,))
    assert mapping.reader_control_disjoint
    assert mapping.geometry_match
    assert mapping.compact()["padding_sides"] == ["right", "left"]
    assert mapping.compact()["fallback_count"] == 0
    assert find_forbidden_fields(mapping.compact()) == []


def test_prompt_token_identity_rejects_serializer_or_tokenizer_drift():
    prompts, tokenized, spans = _valid_two_row_fixture()
    mapping = build_prompt_token_span_map(tokenized, prompts, spans)
    assert_prompt_token_identity(mapping, tokenized, prompts)

    changed_ids = copy.deepcopy(tokenized)
    changed_ids["input_ids"][0][1] += 1
    with pytest.raises(BoundaryError, match="input_ids_identity_changed"):
        assert_prompt_token_identity(mapping, changed_ids, prompts)

    with pytest.raises(BoundaryError, match="prompt_identity_changed"):
        assert_prompt_token_identity(mapping, tokenized, ("changed", prompts[1]))


def test_invalid_row_uses_disabled_zero_anchor_and_preserves_accounting():
    prompts, tokenized, spans = _fixture()
    mapping, decisions = build_prompt_token_span_batch_with_fallback(
        tokenized, prompts, spans, requested_arm="reader"
    )

    assert [decision.effective_arm for decision in decisions] == ["reader", "reader", "disabled"]
    assert [decision.fallback for decision in decisions] == [False, False, True]
    assert decisions[2].reason == "reader_span_out_of_prompt"
    manifest = build_compact_boundary_manifest(mapping, decisions)
    assert manifest["accounting"] == {
        "input_questions": 3,
        "emitted_questions": 3,
        "fallback_questions": 1,
        "unresolved_questions": 0,
    }
    assert manifest["fallback_policy"]["drop_invalid_rows"] is False
    assert manifest["fallback_policy"]["feedback_to_selector"] is False
    assert find_forbidden_fields(manifest) == []


def test_disabled_request_does_not_require_reader_spans():
    prompts, tokenized, spans = _fixture()
    mapping, decisions = build_prompt_token_span_batch_with_fallback(
        tokenized, prompts, spans, requested_arm="disabled"
    )
    assert all(decision.effective_arm == "disabled" for decision in decisions)
    assert all(not decision.fallback for decision in decisions)
    assert all(not row for row in mapping.reader_tokens)


def test_registry_requires_frozen_allowlist_and_exclusive_transport_arm():
    prompts, tokenized, spans = _valid_two_row_fixture()
    mapping = build_prompt_token_span_map(tokenized, prompts, spans)
    registry = ResidualPluginRegistry()
    assert validate_plugin_allowlist(registry.allowlist) == registry.allowlist
    with pytest.raises(BoundaryError, match="plugin_allowlist_not_frozen"):
        validate_plugin_allowlist(tuple(reversed(registry.allowlist)))

    decisions = (
        # The registry must not compose reader and control in one call.
        type("Decision", (), {"requested_arm": "reader", "effective_arm": "reader"})(),
        type("Decision", (), {"requested_arm": "reader", "effective_arm": "control"})(),
    )
    with pytest.raises(BoundaryError, match="mutually_exclusive_arms_required"):
        registry.create_hook(object(), mapping, requested_arm="reader", decisions=decisions)


class _TinyLayer(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states


class _TinyStack(torch.nn.Module):
    def __init__(self, count: int = 32):
        super().__init__()
        self.layers = torch.nn.ModuleList(_TinyLayer() for _ in range(count))


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TinyStack()

    def forward(self, hidden_states):
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def test_hook_is_batch_safe_padding_safe_and_cleans_up():
    prompts, tokenized, spans = _fixture()
    mapping, decisions = build_prompt_token_span_batch_with_fallback(
        tokenized, prompts, spans, requested_arm="reader"
    )
    model = _TinyModel()
    hidden = torch.arange(3 * 8 * 4, dtype=torch.float32).reshape(3, 8, 4)
    baseline = model(hidden.clone())
    layer = model.model.layers[30]

    with ResidualAnchorHook(
        model,
        mapping,
        arms=tuple(decision.effective_arm for decision in decisions),
        alpha=None,
    ) as hook:
        output = model(hidden.clone())
        assert hook.prefill_seen
        assert hook.anchor is not None
        assert torch.equal(output[2], baseline[2])
        assert torch.equal(output[0, 6:], baseline[0, 6:])
        assert torch.equal(output[1, :3], baseline[1, :3])
        assert not torch.equal(output[0, 5], baseline[0, 5])
        assert not torch.equal(output[1, 7], baseline[1, 7])
    assert len(layer._forward_pre_hooks) == 0


def test_disabled_hook_is_exact_noop_and_cleanup_survives_body_exception():
    prompts, tokenized, spans = _fixture()
    mapping, decisions = build_prompt_token_span_batch_with_fallback(
        tokenized, prompts, spans, requested_arm="disabled"
    )
    model = _TinyModel()
    hidden = torch.randn(3, 8, 4)
    baseline = model(hidden.clone())
    layer = model.model.layers[30]
    with ResidualAnchorHook(model, mapping, arms=("disabled",) * 3, alpha=None):
        output = model(hidden.clone())
    assert torch.equal(output, baseline)
    assert len(layer._forward_pre_hooks) == 0

    with pytest.raises(RuntimeError):
        with ResidualAnchorHook(model, mapping, arms=("disabled",) * 3, alpha=None):
            raise RuntimeError("fixture failure")
    assert len(layer._forward_pre_hooks) == 0
