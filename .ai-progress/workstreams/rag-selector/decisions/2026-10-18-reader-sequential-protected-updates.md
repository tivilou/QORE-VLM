# Sequential protected Reader updates as a bounded training-only diagnostic

- Date: 2026-10-18
- Scope: rag-selector
- Status: completed
- Supersedes: None
- Superseded By: None
- Gate: At least two accepted steps with strictly increasing eligible rescue count and zero loss of original or accepted protected members; otherwise stop within fixed budget.
- Authorization: User start 2026-10-18 authorizes implementation and bounded primary CPU training-cache diagnostic; independent utility run and collaborator handoff require a later gate.

## Decision

After one-step feasibility succeeded in both heads, test whether re-linearized shared updates can accumulate rescues while retaining original protected memberships and previously accepted rescues.

## Reason

A one-step witness exists, but quantum-final repair can abandon a prior rescue. A sequential guard distinguishes accumulable rescue from isolated near-tie feasibility.

## Consequences

Use initial quantum/classical heads, fixed candidate order and matched CPU budget; no evaluation100/87, no Generator, no full-data or collaborator run. Native all59 membership is the acceptance gate; success remains L0.

## Outcome

Run `20261018T045707538341Z` passed the frozen gate. Both initial heads accumulated three strict rescues with zero original-protected loss and zero loss of previously accepted rescues. Quantum required 8 candidate attempts and classical 5; both shared the first two rescued cases, then diverged on the third. Independent replay reproduced all 27 native all59 trials (6 accepted, 21 rejected) and the complete 378-array trace. The result establishes accumulable training feasibility, not held-out utility or quantum superiority.

## Next constraint

Do not expand the rescue count on these 59 fitted questions. The next experiment must freeze a genuinely unseen question cohort before fitting, learn updates using training59 only, and evaluate direct/useful Top5 membership on the untouched cohort with an identity Reader and matched classical control. It requires separate authorization and remains a new task.
