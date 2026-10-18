# Untouched validation87 transport preflight before semantic review

- Date: 2026-10-18
- Scope: rag-selector
- Status: closed
- Supersedes: None
- Superseded By: None
- Gate: Advance to blinded semantic review only if the frozen quantum arm changes Top5 membership on at least 5/87 questions and differs from the matched classical arm on at least 3/87 questions.
- Authorization: User start 2026-10-18 authorizes implementation and a bounded label-blind primary-server preflight; no semantic labeling or utility claim is authorized by this gate.

## Decision

Use the 87 future-role validation questions as an untouched transport cohort. First run the already-frozen DPR Reader on their existing Top50 question/passage inputs, verify Reader score and Top5 parity, and then apply the final quantum and matched classical heads from the completed training59 sequential run exactly once. Do not fit, tune, inspect answers, consume support annotations, call the Generator, or retrieve new passages.

## Why this ordering is necessary

The cohort audit found 87 validation rows, but all 87 remain pending: there are no qualified support labels and no eligible/protected target indices. Therefore a direct/useful membership result cannot be computed yet. A label-blind transport gate distinguishes “the learned heads actually move unseen selections” from “there is no unseen effect” without leaking labels or spending a large blinded review prematurely.

## Frozen contract

- Source bundle: `f723bbb6f9075c3e04fd3d8a7b2d3f8127b6e354422ee2a2f6e167bfe32248ce`.
- Population: exactly 87 validation questions × 50 already stored candidates.
- Reader: `facebook/dpr-reader-single-nq-base`, revision `38f47a4986084c53447ba92ab0a83076b58d86a8`, local files only, max length 350, batch size 8.
- Sequential values: run `20261018T045707538341Z`, SHA256 `35d7f9c165c581e7ce21ccaf57eca104db6e97a2b2af6c598af333caac8287bc`.
- Arms: identity Reader, frozen quantum final head, frozen matched classical final head.
- The transport threshold is a review-effort gate, not a performance gate. Even a pass remains L0 until qualified blinded evidence exists.

## Consequences

The feature cache may contain only numeric CLS/pooled/Reader-score arrays and question/candidate digests. The transport trace may contain only score vectors, Top5 indices, hashes and set-change metrics. If the gate passes, a separate score/rank/method-blind review packet is required; if it fails, this sequential update family closes without claiming unseen utility. Quantum-specific attribution still requires a matched classical comparison and cannot be inferred from set movement alone.

## Outcome

Run `20261018T073840422368Z` passed the two review-effort thresholds: the frozen quantum head changed `13/87` Top5 sets and differed from the matched classical head on `9/87`; the classical head changed `9/87`. Independent replay verified all261 score vectors, all87 Top5 sets and the unopened label boundary. This is L0 transport evidence only. The next stage is a separately frozen two-reviewer score/rank/method-blind semantic audit over the15-case union where any arm differs.
