# Phase 9H Answerability and Generation Diagnostic

Phase 9G found that complete answer-bearing passages, answer-passage ordering,
and an answer-bearing Top-50 oracle explain only a small minority of selected-hit
baseline errors. Phase 9H therefore keeps retrieval and QORE selection frozen
and separates the residual errors into evidence support, answer extraction,
answer form, and unresolved semantic generation signals.

The runner uses fresh `nq_open validation[800:850]`. It retrieves Top-50 and
selects QORE K=5 exactly once. Diagnostic arms run only for selected-hit
baseline EM errors with a successful deterministic gold-answer copy control.
The extractive arm asks the same pinned generator to copy an answer span from
the selected context. The support-judge arm asks the same generator whether
the internal gold answer is directly supported by the selected context and
records only `supported`, `unsupported`, or `uncertain`; this is explicitly a
model signal, not ground truth. Baseline F1 supplies a deterministic answer-form
signal without exposing prediction text.

Attribution is report-only and hierarchical. Explicitly unsupported support
judgments are separated from extraction recovery, positive baseline F1 is
reported as a possible answer-form mismatch, and contradictory support/extraction
signals are kept as `conflicting_signals`. No arm output can affect selection,
retrieval, generation settings, labels, or evaluation.
