"""Deterministic question decomposition used by the Q-DES probe.

The parser is deliberately conservative. It never invents a ``what`` slot
when a question cannot be parsed, and it never reads answers, evidence labels,
retrieval output, or generated text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "in", "is", "it",
    "its", "of", "on", "or", "that", "the", "their", "there", "these", "this",
    "to", "was", "were", "what", "when", "where", "which", "who", "whom", "whose",
    "why", "how", "would", "with", "about", "into", "during", "after", "before",
    "than", "then", "they", "them", "he", "she", "his", "her", "you", "your",
}


@dataclass(frozen=True)
class QuestionSlots:
    """Question-only typed slots."""

    slots: Dict[str, str]
    success: bool
    parser_confidence: float


class QuestionSlotParser:
    """Parse a question into operator, answer type, subject, and relation."""

    def parse(self, question: str) -> QuestionSlots:
        if not isinstance(question, str):
            return QuestionSlots({}, False, 0.0)

        text = question.strip().lower()
        tokens = _TOKEN_RE.findall(text)
        if len(tokens) < 2:
            return QuestionSlots({}, False, 0.0)

        operator, answer_type = self._detect_operator_and_type(text, tokens)
        content = [token for token in tokens if token not in _STOPWORDS]
        if not operator or not content:
            return QuestionSlots({}, False, 0.0)

        # Keep a short anchor and the complete content relation separately.
        slots = {
            "operator": operator,
            "answer_type": answer_type,
            "subject": " ".join(content[: min(4, len(content))]),
            "relation": " ".join(content),
        }

        confidence = 0.45
        confidence += 0.15 if operator else 0.0
        confidence += 0.15 if answer_type else 0.0
        confidence += 0.15 if len(content) >= 2 else 0.0
        confidence += 0.10 if text.endswith("?") else 0.0
        confidence = min(1.0, confidence)
        # There is intentionally no unconditional fallback slot.
        success = bool(operator and answer_type and len(content) >= 1)
        return QuestionSlots(slots, success, confidence if success else 0.0)

    def _detect_operator_and_type(self, text: str, tokens: List[str]) -> tuple[str, str]:
        if text.startswith("how many") or text.startswith("how much"):
            return "count", "quantity"
        first = tokens[0]
        if first in {"who", "whom", "whose"}:
            return first, "person"
        if first == "where":
            return first, "location"
        if first == "when":
            return first, "date"
        if first == "why":
            return first, "explanation"
        if first == "how":
            return first, "procedure"
        if first == "which":
            return first, "entity"
        if first == "what":
            if len(tokens) > 1 and tokens[1] in {"year", "date", "time"}:
                return first, "date"
            if len(tokens) > 1 and tokens[1] in {"country", "city", "state", "place", "location"}:
                return first, "location"
            return first, "entity"
        return "", ""


_parser: QuestionSlotParser | None = None


def get_parser() -> QuestionSlotParser:
    """Return the process-local deterministic parser."""

    global _parser
    if _parser is None:
        _parser = QuestionSlotParser()
    return _parser
