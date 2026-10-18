# Two-reviewer blind audit for unseen Top5 transport

- Date: 2026-10-18
- Scope: rag-selector
- Status: active
- Supersedes: None
- Superseded By: None
- Gate: after zero unresolved blind-review disagreements, quantum gains at least2 useful items over Reader, loses no direct items, gains at least1 useful item over matched classical, has at least one more case win than loss versus Reader, and has zero uncertain selected items
- Authorization: packet construction and blind-review collection only; method unblinding is authorized only after complete consensus, and no L1/L2 or quantum-advantage claim is authorized

## Decision

Review the union of all identity Reader, frozen quantum and frozen matched-classical Top5 passages for the15 validation cases where at least two sets differ. The reviewer view contains only opaque case/item IDs, the question, title and passage text. It excludes answers, candidate identity, retrieval/Reader ranks, every score, method identity and selection membership.

Use two independent reviews with `answer_support` (`direct`, `partial`, `none`, `uncertain`) and `scope_match` (`yes`, `partial`, `no`, `uncertain`). Merge semantic codes while still blind. Any disagreement produces an opaque dispute packet and blocks the sealed method key. Only complete consensus can be scored with the prospective gate.

## Reason

The transport result proves that the fitted heads move unseen selections but cannot establish whether the moves help. Reviewing only the91 passages in changed-set unions is sufficient because the other72 questions have identical selections and therefore cancel in pairwise comparison. Separating review from the method key prevents score, rank and method expectations from contaminating semantic judgments.

## Consequences

The packet is L0 and diagnostic. Passing its gate would justify a larger untouched-data utility study, not L1/L2 or quantum advantage. Failing the gate closes this fitted sequential-update candidate without retuning on validation87. Reviewer materials and the sealed key must remain separate until consensus.
