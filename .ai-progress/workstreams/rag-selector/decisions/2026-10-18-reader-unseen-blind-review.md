# Two-reviewer blind audit for unseen Top5 transport

- Date: 2026-10-18
- Scope: rag-selector
- Status: active
- Supersedes: None
- Superseded By: None
- Gate: after zero unresolved blind-review disagreements, quantum gains at least2 useful items over Reader, loses no direct items, gains at least1 useful item over matched classical, has at least one more case win than loss versus Reader, and has zero uncertain selected items
- Authorization: design and preregistration of a larger untouched utility study are authorized; no larger run, L1/L2 claim, or quantum-advantage claim is authorized until its fresh cohort and gate are frozen

## Decision

Review the union of all identity Reader, frozen quantum and frozen matched-classical Top5 passages for the15 validation cases where at least two sets differ. The reviewer view contains only opaque case/item IDs, the question, title and passage text. It excludes answers, candidate identity, retrieval/Reader ranks, every score, method identity and selection membership.

Use two independent reviews with `answer_support` (`direct`, `partial`, `none`, `uncertain`) and `scope_match` (`yes`, `partial`, `no`, `uncertain`). Merge semantic codes while still blind. Any disagreement produces an opaque dispute packet and blocks the sealed method key. Only complete consensus can be scored with the prospective gate.

## Reason

The transport result proves that the fitted heads move unseen selections but cannot establish whether the moves help. Reviewing only the91 passages in changed-set unions is sufficient because the other72 questions have identical selections and therefore cancel in pairwise comparison. Separating review from the method key prevents score, rank and method expectations from contaminating semantic judgments.

## Consequences

Both isolated reviews covered91/91 items. They agreed on85 semantic-code pairs; a third blind adjudication resolved six disputes before method unblinding. On the15 changed cases, Reader / quantum / classical retained useful evidence `38 / 41 / 40` and direct evidence `25 / 25 / 25`. Quantum was `4/1/10` wins/losses/ties versus Reader with point delta `+3`, and `2/1/12` versus classical with delta `+1`. All five frozen checks passed with zero uncertain selected items.

The result advances the candidate to design of a larger untouched utility study. It remains L0 because72/87 cases did not change, only15 changed cases contributed comparative information, the semantic sample is small, and the quantum and classical arms differ by only one useful item. Validation87 must not be retuned or reused as fresh evidence. The next cohort must be identity-disjoint from train59, validation87 and the repeatedly inspected100-case panel, and its contract must be frozen before execution.
