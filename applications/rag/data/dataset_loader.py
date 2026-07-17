"""Dataset loader for RAG evaluation.

Loads QA datasets and normalizes them to a common format:
    {
        'id': str | int,
        'question': str,
        'answers': list[str],
        'gold_passages': list[dict] (optional, for aligned/faiss gold tracking)
    }

Supports:
- Natural Questions (nq_open)
- HotpotQA (distractor and fullwiki)
- Custom JSON/JSONL

Each dataset is adapted via a loader function that handles its schema quirks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def load_nq_open(split: str = "validation", max_samples: int = 0) -> list[dict]:
    """Load Natural Questions Open (nq_open from HuggingFace).

    Returns questions with 'id', 'question', 'answers'. Gold passages are NOT
    included in nq_open — for gold-aligned eval, you need to resolve them from
    the full NQ dataset or wiki_dpr separately (see `resolve_nq_gold`).
    """
    from datasets import load_dataset

    ds = load_dataset("nq_open", split=split)
    samples = []
    for i, item in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        samples.append({
            "id": f"nq_{i}",
            "question": item["question"],
            "answers": item["answer"],  # already a list in nq_open
        })
    return samples


def load_hotpotqa_distractor(split: str = "validation", max_samples: int = 0) -> list[dict]:
    """Load HotpotQA distractor setting.

    Each question has 10 candidate passages (2 gold + 8 distractors). We flatten
    context into a 'candidates' list and record 'gold_local_indices' pointing to
    the gold within that list. This is the natural format for precomputed mode.
    """
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)
    samples = []
    for i, item in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        # HotpotQA context: {'title': [...], 'sentences': [[...], [...]]}
        titles = item["context"]["title"]
        sentences_list = item["context"]["sentences"]
        candidates = []
        for title, sents in zip(titles, sentences_list):
            text = " ".join(sents)
            candidates.append({"title": title, "text": text})

        # Gold passage titles from supporting_facts
        gold_titles = set(item["supporting_facts"]["title"])
        gold_local_indices = [
            j for j, c in enumerate(candidates) if c["title"] in gold_titles
        ]

        samples.append({
            "id": item["id"],
            "question": item["question"],
            "answers": [item["answer"]],
            "candidates": candidates,
            "gold_local_indices": gold_local_indices,
        })
    return samples


def load_hotpotqa_fullwiki(split: str = "validation", max_samples: int = 0) -> list[dict]:
    """Load HotpotQA fullwiki setting (no precomputed candidates).

    Returns just question/answer/gold passage titles. Retrieval is required.
    """
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "fullwiki", split=split, trust_remote_code=True)
    samples = []
    for i, item in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        gold_titles = set(item["supporting_facts"]["title"])
        samples.append({
            "id": item["id"],
            "question": item["question"],
            "answers": [item["answer"]],
            "gold_passage_titles": gold_titles,
        })
    return samples


def load_jsonl(path: str, max_samples: int = 0) -> list[dict]:
    """Load custom JSONL where each line is a question dict.

    Expected keys: 'id'?, 'question', 'answers' (or 'answer').
    Optional: 'gold_passages', 'candidates', etc.
    """
    import json

    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            # Normalize 'answer' -> 'answers'
            if "answer" in item and "answers" not in item:
                ans = item["answer"]
                item["answers"] = [ans] if isinstance(ans, str) else ans
            if "id" not in item:
                item["id"] = f"custom_{i}"
            samples.append(item)
    return samples


def load_dataset_for_rag(
    dataset_name: str,
    split: str = "validation",
    max_samples: int = 0,
    custom_path: Optional[str] = None,
) -> list[dict]:
    """Unified loader: dispatch by dataset_name.

    Args:
        dataset_name: "nq_open", "hotpotqa_distractor", "hotpotqa_fullwiki", "jsonl".
        split: dataset split (ignored for jsonl).
        max_samples: limit number of samples (0 = all).
        custom_path: path for jsonl mode.
    """
    name = dataset_name.lower()
    # Short aliases used by eval_suite.py / eval_rag_refactored.py CLIs.
    if name in ("nq", "nq_open", "natural_questions"):
        return load_nq_open(split, max_samples)
    if name in ("hotpotqa", "hotpotqa_distractor"):
        return load_hotpotqa_distractor(split, max_samples)
    if name == "hotpotqa_fullwiki":
        return load_hotpotqa_fullwiki(split, max_samples)
    if name == "jsonl":
        if not custom_path:
            raise ValueError("jsonl mode requires custom_path")
        return load_jsonl(custom_path, max_samples)
    raise ValueError(f"Unknown dataset: {dataset_name}")
