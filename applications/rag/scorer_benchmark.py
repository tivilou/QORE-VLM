"""Classical answer-scorer registry for the fixed Top-50 benchmark.

The registry is deliberately independent from the production selector.  A
loaded scorer only receives a question and candidate passage text and returns
one finite score per candidate.  Selection, evidence labels, answers, and
evaluation metrics stay outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


class ScorerBenchmarkError(RuntimeError):
    """Raised when a scorer violates the benchmark contract."""


@dataclass(frozen=True)
class ScorerSpec:
    """Pinned identity and loading contract for one scorer."""

    scorer_id: str
    backend: str
    model_id: str
    revision: str | None = None
    max_length: int = 512
    trust_remote_code: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "scorer_id": self.scorer_id,
            "backend": self.backend,
            "model_id": self.model_id,
            "revision": self.revision,
            "max_length": self.max_length,
            "trust_remote_code": self.trust_remote_code,
        }


# The first round is intentionally small and heterogeneous: two NQ-trained
# DPR readers, two modern cross-encoder rerankers, and one multilingual
# reranker.  Exact resolved commit hashes are emitted after loading.
DEFAULT_SCORER_SPECS: tuple[ScorerSpec, ...] = (
    ScorerSpec(
        "dpr_reader_single_nq",
        "dpr_reader",
        "facebook/dpr-reader-single-nq-base",
        max_length=350,
    ),
    ScorerSpec(
        "dpr_reader_multiset",
        "dpr_reader",
        "facebook/dpr-reader-multiset-base",
        max_length=350,
    ),
    ScorerSpec(
        "bge_reranker_large",
        "hf_sequence_classifier",
        "BAAI/bge-reranker-large",
        max_length=512,
    ),
    ScorerSpec(
        "bge_reranker_v2_m3",
        "hf_sequence_classifier",
        "BAAI/bge-reranker-v2-m3",
        max_length=512,
    ),
    ScorerSpec(
        "jina_reranker_v2_base_multilingual",
        "hf_sequence_classifier",
        "jinaai/jina-reranker-v2-base-multilingual",
        max_length=1024,
        trust_remote_code=True,
    ),
)


def scorer_specs_by_id() -> dict[str, ScorerSpec]:
    return {spec.scorer_id: spec for spec in DEFAULT_SCORER_SPECS}


def _finite_scores(values: Any, expected: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (expected,):
        raise ScorerBenchmarkError(
            f"scorer returned {array.shape[0]} scores for {expected} candidates"
        )
    if not np.all(np.isfinite(array)):
        raise ScorerBenchmarkError("scorer returned a non-finite score")
    return array


def _config_hash(config: Any) -> str | None:
    try:
        if hasattr(config, "to_dict"):
            payload = config.to_dict()
        elif isinstance(config, Mapping):
            payload = dict(config)
        else:
            return None
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded).hexdigest()


class LoadedScorer:
    """Thin, auditable wrapper around one locally loaded HF scorer."""

    def __init__(self, spec: ScorerSpec, *, device: str | None = None, batch_size: int = 16):
        self.spec = spec
        self.batch_size = int(batch_size)
        self._device_name = device
        self._backend_kind = spec.backend
        self._load()

    def _load(self) -> None:
        if self.spec.backend == "dpr_reader":
            self._load_dpr()
        elif self.spec.backend == "hf_sequence_classifier":
            self._load_sequence_classifier()
        else:
            raise ScorerBenchmarkError(f"unknown scorer backend: {self.spec.backend}")

    def _torch_device(self, torch: Any) -> Any:
        requested = self._device_name
        if requested:
            return torch.device(requested)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_dpr(self) -> None:
        import torch
        from transformers import DPRReader, DPRReaderTokenizer

        kwargs = {"revision": self.spec.revision} if self.spec.revision else {}
        self.tokenizer = DPRReaderTokenizer.from_pretrained(self.spec.model_id, **kwargs)
        self.model = DPRReader.from_pretrained(self.spec.model_id, **kwargs)
        self.device = self._torch_device(torch)
        self.model.to(self.device)
        self.model.eval()
        self._resolved_revision = getattr(self.model.config, "_commit_hash", None) or self.spec.revision
        self._config_sha256 = _config_hash(self.model.config)

    def _load_sequence_classifier(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        kwargs = {"revision": self.spec.revision} if self.spec.revision else {}
        kwargs["trust_remote_code"] = bool(self.spec.trust_remote_code)
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.spec.model_id, **kwargs)
        self.device = self._torch_device(torch)
        self.model.to(self.device)
        self.model.eval()
        self._resolved_revision = getattr(self.model.config, "_commit_hash", None) or self.spec.revision
        self._config_sha256 = _config_hash(self.model.config)

    def score(self, question: str, passages: Sequence[str]) -> np.ndarray:
        """Return one raw relevance score per passage, preserving input order."""

        if not isinstance(question, str) or not question.strip():
            raise ScorerBenchmarkError("question must be a non-empty string")
        texts = list(passages)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ScorerBenchmarkError("passages must contain non-empty strings")
        if not texts:
            return np.empty(0, dtype=np.float64)
        import torch

        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                if self.spec.backend == "dpr_reader":
                    encoded = self.tokenizer(
                        questions=[question] * len(batch),
                        texts=batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.spec.max_length,
                    )
                    encoded = {key: value.to(self.device) for key, value in encoded.items()}
                    logits = self.model(**encoded).relevance_logits
                else:
                    encoded = self.tokenizer(
                        [question] * len(batch),
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.spec.max_length,
                    )
                    encoded = {key: value.to(self.device) for key, value in encoded.items()}
                    logits = self.model(**encoded).logits
                    if logits.ndim == 2:
                        logits = logits[:, -1]
                    else:
                        logits = logits.reshape(-1)
                chunks.append(logits.detach().float().cpu().numpy())
        return _finite_scores(np.concatenate(chunks), len(texts))

    def identity(self) -> dict[str, Any]:
        return {
            **self.spec.as_dict(),
            "resolved_revision": self._resolved_revision,
            "config_sha256": self._config_sha256,
            "device": str(self.device),
            "backend_runtime": self._backend_kind,
            "batch_size": self.batch_size,
        }


def build_scorer(spec: ScorerSpec, *, device: str | None = None, batch_size: int = 16) -> LoadedScorer:
    """Construct one allowlisted scorer; no filesystem discovery is used."""

    return LoadedScorer(spec, device=device, batch_size=batch_size)


def rank_indices(scores: Sequence[float], ranks: Sequence[int], identifiers: Sequence[str], k: int) -> list[int]:
    """Stable descending score order with rank/id tie breaks."""

    values = _finite_scores(scores, len(ranks))
    if len(identifiers) != len(ranks):
        raise ScorerBenchmarkError("identifier/rank length mismatch")
    if k < 1 or k > len(values):
        raise ScorerBenchmarkError(f"invalid k={k} for {len(values)} scores")
    order = sorted(
        range(len(values)),
        key=lambda index: (-float(values[index]), int(ranks[index]), str(identifiers[index])),
    )
    return order[:k]


def score_summary(scores: Sequence[float]) -> dict[str, float]:
    values = _finite_scores(scores, len(scores))
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }

