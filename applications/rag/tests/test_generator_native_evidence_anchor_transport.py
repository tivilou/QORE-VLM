from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from applications.rag.generator_native_evidence_anchor_transport import (
    BoundaryError,
    FrozenDPRReaderSpanProvider,
    GeneratorNativeEvidenceAnchorObserver,
    GeneratorPromptSpanAdapter,
    ResidualAnchorHook,
    ReaderSpanProvenance,
    ResidualPluginRegistry,
    assert_prompt_token_identity,
    build_compact_boundary_manifest,
    build_prompt_token_span_batch_with_fallback,
    build_prompt_token_span_map,
    find_forbidden_fields,
    validate_plugin_allowlist,
)


class _FakeReaderScorer:
    def score_passages_with_hypotheses(self, question, passages, *, top_m, max_answer_tokens):
        assert question == "Which color?"
        assert top_m == 3
        assert max_answer_tokens == 10
        return [0.9] * len(passages), [
            [{"text": "blue", "normalized": "blue", "probability": 1.0}],
            [{"text": "blue", "normalized": "blue", "probability": 1.0}],
        ]


class _FakeTokenizer:
    eos_token_id = 0
    all_special_ids = ()

    def __call__(self, prompt, *, return_tensors=None, truncation=False, max_length=None,
                 return_offsets_mapping=False, return_special_tokens_mask=False, **kwargs):
        tokens = []
        offsets = []
        for match in __import__("re").finditer(r"\S+", prompt):
            tokens.append(sum((index + 1) * ord(char) for index, char in enumerate(match.group(0))) % 10000 + 1)
            offsets.append([match.start(), match.end()])
        if max_length is not None:
            tokens = tokens[:max_length]
            offsets = offsets[:max_length]
        import torch
        output = {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.ones((1, len(tokens)), dtype=torch.long),
        }
        if return_offsets_mapping:
            output["offset_mapping"] = torch.tensor([offsets], dtype=torch.long)
        if return_special_tokens_mask:
            output["special_tokens_mask"] = torch.zeros((1, len(tokens)), dtype=torch.long)
        return output


class _FakeGenerator:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.model = _TinyModel()

    def _build_prompt(self, question, passages):
        context = "\n\n".join(f"Passage {i + 1}: {p}" for i, p in enumerate(passages))
        return f"Context: {context} Question: {question} Answer:"

    def generate(self, question, passages):
        import torch
        encoded = self.tokenizer(
            self._build_prompt(question, passages),
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        hidden = torch.zeros((1, len(encoded["input_ids"][0]), 4))
        self.model(hidden)
        return "stable"


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


def test_alpha_must_follow_architecture_rule():
    prompts, tokenized, spans = _valid_two_row_fixture()
    mapping = build_prompt_token_span_map(tokenized, prompts, spans)
    with pytest.raises(BoundaryError, match="alpha_rule_violation"):
        with ResidualAnchorHook(_TinyModel(), mapping, arms=("reader", "reader"), alpha=0.5):
            pass


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


def test_gold_free_provider_locates_unique_hypothesis_and_rejects_ambiguous_text():
    provider = FrozenDPRReaderSpanProvider(_FakeReaderScorer())
    result = provider.provide("Which color?", ("The sky is blue.", "blue blue"))
    assert result.provider_status == "ok"
    assert result.passage_spans == (((11, 15),), ())
    assert result.hypothesis_count == 2
    assert result.located_count == 1
    assert result.rejected_count == 1
    assert result.compact()["gold_used"] is False


def test_prompt_adapter_uses_unchanged_generator_serializer_and_observer_fallback():
    generator = _FakeGenerator()
    provider = FrozenDPRReaderSpanProvider(_FakeReaderScorer())
    observer = GeneratorNativeEvidenceAnchorObserver(generator, provider)
    observation = observer.generate(
        "Which color?", ("The sky is blue.", "blue blue"), requested_arm="reader"
    )
    assert observation.text == "stable"
    assert observation.decision.effective_arm == "reader"
    assert observation.decision.fallback is False
    assert observation.mapping is not None
    assert observation.hook_prefill_seen is True
    assert observation.hook_call_count >= 1
    compact = observation.compact()
    assert compact["raw_content_persisted"] is False
    assert find_forbidden_fields(compact) == []


def test_prompt_adapter_rejects_serializer_drift_before_hook_and_runs_disabled_generation():
    class DriftingGenerator(_FakeGenerator):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def _build_prompt(self, question, passages):
            self.calls += 1
            suffix = str(self.calls)
            context = "\n\n".join(f"Passage {i + 1}: {p}" for i, p in enumerate(passages))
            return f"Context: {context} Question: {question} Answer:{suffix}"

    generator = DriftingGenerator()
    observer = GeneratorNativeEvidenceAnchorObserver(generator, FrozenDPRReaderSpanProvider(_FakeReaderScorer()))
    observation = observer.generate("Which color?", ("The sky is blue.", "blue blue"), requested_arm="reader")
    assert observation.decision.effective_arm == "disabled"
    assert observation.decision.fallback is True
    assert observation.decision.reason == "prompt_serializer_nondeterministic"
    assert observation.boundary_status == "unwrapped_disabled_fallback"


def test_observer_does_not_retry_generation_after_hook_installation_failure():
    class FailingGenerator(_FakeGenerator):
        def __init__(self):
            super().__init__()
            self.generate_calls = 0

        def generate(self, question, passages):
            self.generate_calls += 1
            prompt = self._build_prompt(question, passages)
            encoded = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=4096
            )
            hidden = torch.zeros((1, len(encoded["input_ids"][0]), 4))
            self.model(hidden)
            raise RuntimeError("generation failed")

    generator = FailingGenerator()
    layer = generator.model.model.layers[30]
    observer = GeneratorNativeEvidenceAnchorObserver(
        generator, FrozenDPRReaderSpanProvider(_FakeReaderScorer())
    )
    with pytest.raises(RuntimeError, match="generation failed"):
        observer.generate(
            "Which color?", ("The sky is blue.", "blue blue"), requested_arm="reader"
        )
    assert generator.generate_calls == 1
    assert len(layer._forward_pre_hooks) == 0


def test_cached_llama_tokenizer_maps_the_frozen_generator_chat_prompt_without_drift():
    revision = "53346005fb0ef11d3b6a83b12c895cca40156b6c"
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B-Instruct"
        / "snapshots"
        / revision
    )
    if not (snapshot / "config.json").is_file():
        pytest.skip("pinned Llama tokenizer is not cached")
    from transformers import AutoTokenizer
    from applications.rag.generation.generator import Generator

    generator = Generator.__new__(Generator)
    generator.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    generator.use_chat_template = True
    passages = ("The synthetic marker is cobalt.", "A separate neutral passage has context.")
    provenance = ReaderSpanProvenance(
        passage_spans=(((24, 30),), ()),
        hypothesis_count=1,
        located_count=1,
        rejected_count=0,
        provider_status="ok",
    )
    mapping, decisions = GeneratorPromptSpanAdapter().build(
        generator,
        "Which synthetic marker is named?",
        passages,
        provenance,
        requested_arm="reader",
    )
    assert decisions[0].effective_arm == "reader"
    assert decisions[0].fallback is False
    assert mapping.reader_tokens[0]
    assert mapping.geometry_match
    assert mapping.reader_control_disjoint
