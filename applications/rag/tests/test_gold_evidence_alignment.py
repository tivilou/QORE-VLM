from applications.rag.gold_evidence_alignment import extract_gold_evidence


def test_extracts_columnar_hf_annotation_and_tokens():
    record = {
        "question": {"text": "Who won?"},
        "document": {
            "title": "Example Page",
            "tokens": {
                "token": ["Noise", "Ada", "Lovelace", "tail"],
                "is_html": [False, False, False, False],
            },
        },
        "annotations": {
            "short_answers": {
                "start_token": [[1]],
                "end_token": [[3]],
                "text": [[""]],
            }
        },
    }

    evidence = extract_gold_evidence(record, fallback_answers=["fallback"])

    assert evidence.document_title == "Example Page"
    assert evidence.strict_answers == ("ada lovelace",)
    assert evidence.support_answers == ("ada lovelace", "fallback")
    assert evidence.has_short_span is True
    assert evidence.source == "nq_annotations"


def test_extracts_multiple_columnar_annotations_deterministically():
    record = {
        "question": {"text": "Who?"},
        "document": {
            "title": "Example Page",
            "tokens": {
                "token": ["Ada", "Lovelace", "or", "Charles", "Babbage"],
                "is_html": [False, False, False, False, False],
            },
        },
        "annotations": {
            "short_answers": {
                "start_token": [0, 3],
                "end_token": [2, 5],
                "text": ["Ada Lovelace", ""],
            }
        },
    }

    evidence = extract_gold_evidence(record)

    assert evidence.strict_answers == ("ada lovelace", "charles babbage")
    assert evidence.has_short_span is True
