# Phase 9G Context Failure Localization

Phase 9F found a dominant `beyond_local_context` class. Phase 9G is a held-out,
observation-only diagnostic that separates three narrower explanations without
changing QORE selection, retrieval, generation, evaluation, or labels.

The runner retrieves Top-50 and selects QORE K=5 exactly once. It runs the
diagnostic arms only when the selected context contains a gold answer, the
baseline has EM=0, and the deterministic gold-answer copy control has EM=1.
The arms are: all complete answer-bearing selected passages, the same passages
with existing answer spans marked, all selected passages with answer-bearing
passages moved first, and all answer-bearing Top-50 passages in retrieval order.
Raw contexts and predictions never cross the compact output boundary.

Attribution is exclusive and follows the weakest successful intervention. A
successful plain full-passage arm means the local Phase 9F window omitted useful
within-passage context. Otherwise highlighted full passage indicates
localization, answer-first indicates ordering/attention only when it changes
order, and the Top-50 oracle indicates additional answer-bearing passages only
when it adds such passages. A successful nonincremental oracle is reported
separately. No success is `beyond_top50_context`. Every outcome is report-only.
A dominant class does not authorize a production selector or generator change.
