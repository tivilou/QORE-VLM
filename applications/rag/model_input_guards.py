"""Input-integrity checks shared by the pinned DPR and BGE adapters (stdlib).

Pair truncation is allowed to shorten a passage. It must never silently change
the question. This is a rejection guard, not a new truncation/fallback policy.
"""


def require_complete_question(tokenizer, question, input_ids):
    """Verify the original first-sequence tokens before any model forward.

    Both supported native tokenizers start pairs with CLS, question, SEP.
    Compare the full expected prefix, rather than finding the first SEP: the
    question itself can contain a literal special token. No text/IDs are logged.
    """
    question_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(question))
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    if not question_ids or type(cls) is not int or type(sep) is not int:
        raise ValueError('Question prefix cannot be certified for the pinned tokenizer')
    prefix = [cls, *question_ids, sep]
    rows = input_ids.tolist() if hasattr(input_ids, 'tolist') else input_ids
    if not rows:
        raise ValueError('Question prefix requires a nonempty pair batch')
    for index, ids in enumerate(rows):
        if list(ids[:len(prefix)]) != prefix:
            raise ValueError(
                f'QUESTION_TOKENS_TRUNCATED_OR_CHANGED at batch row {index}; '
                'stop before model forward; owner must qualify the common input '
                'or explicitly implement a shared whole-question retrieval fallback')
