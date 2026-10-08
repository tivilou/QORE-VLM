# Separate fixed Silver membership from useful-evidence retention

- Date: 2026-10-08
- Scope: rag-selector
- Status: active
- Supersedes: None
- Superseded By: None
- Gate: declare metric semantics and target-exposure boundaries before the next experiment
- Authorization: offline audit and progress recording; no prospective experiment implemented

## Decision

Report exact Silver-set overlap separately from useful/direct panel evidence retention. The fixed five-member reference includes nonpositive fillers in this panel, so a membership loss is not necessarily evidence loss and an alternative positive need not be in the reference. Future screen designs must prospectively specify these distinct metrics, eligible-question handling, and independent confirmation. Do not retroactively replace the earlier preregistered gate or claim L1/L2 from Silver diagnostics.

## Evidence and correction

The verified reference attains per-question positive/direct count ceilings of 307/222 selected slots overall, but includes 170 irrelevant and 23 uncertain fillers. Sixty-one questions have fewer than five positive candidates. Q-ARCG versus Reader is 349/355 exact memberships, **204/203 positive-consensus memberships**, **138/138 direct**, and **171/171 all-three-positive**. The supplementary positive gain of one passage is statistically inconclusive. Six of the eight nominal overlap losses leave positive/direct retention unchanged.

See [all-case/changed-case audit](../refs/qarcg_screen_20261007_semantic_case_audit.md). This adds an explicit correction to the earlier audit's interpretation without rewriting its numerical results.

## Next-action rationale

Keep the original relevance anchor, which avoided the fixed-fusion family's large degradation. Inspect existing training provenance first; do not rescue the current candidate by tuning coefficients on these 100 questions. The provisional next quantum direction changes the information entering the head: question-conditioned Reader semantic representations rather than four summary scalars. Its goal is relation-support discrimination, not indiscriminately promoting sharp spans or imitating arbitrary fillers. This remains exploratory pending training audit, a qualified design, independent validation, novelty checks, and a matched classical control.
