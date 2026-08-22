# Phase 9I Candidate Coverage and Grounded Ranking Diagnostic

## Authorization and boundary

This is a plan-only, observation-only follow-up authorized after the Phase 9H
mixed gate. It does not implement a selector, Generator, evaluator, label, or
retrieval change. The existing QORE K=5 path remains the frozen reference.

The only new boundary is after one unchanged retrieval and one unchanged QORE
selection. Runtime candidate strings and verifier outputs stay in memory. GitHub
receives only compact scalar summaries and provenance metadata.

## Question being separated

For a selected-hit question on which the frozen baseline has EM=0, there are
two distinct possibilities:

1. no correct answer is present in the small candidate set supported by the
   selected context; or
2. a correct candidate is present, but a terminal choice rule fails to select it.

The diagnostic measures both rather than treating a single prompt arm as a
causal explanation.

## Candidate set

The pre-registered set has at most three candidates:

- baseline_v1: the unchanged Generator.generate output;
- extractive_span_v1: a deterministic prompt that copies a shortest answer
  span from the selected context;
- evidence_constrained_v1: a deterministic prompt that returns a short answer
  directly supported by the selected context, or <ABSTAIN>.

All candidate producers see only the question and selected context. They do not
see gold, labels, EM/F1, evaluator output, QORE scores, or retrieval diagnostics.
Candidates are normalized only for duplicate counting; raw candidate strings are
not persisted.

## Eligibility

The candidate probes are conditionally run only for the offline diagnostic cohort:

- selected context has a lexical gold match;
- baseline EM is zero;
- the gold-copy control succeeds.

The gold-copy control is a diagnostic control, not an inference input. Questions
outside this cohort retain the baseline-only record and do not enter the
candidate oracle denominator.

## Verifier

The verifier receives the question, selected context, candidate mode IDs, and
runtime candidate strings. It returns only a candidate mode or index. It is run
twice: once in the declared candidate order and once under a fixed permutation.
The audit records agreement and position sensitivity without using gold.

Only after both calls finish does the unchanged evaluator attach EM/F1 to the
selected candidate and the candidate oracle.

## Metrics and gates

The primary coverage statistic is the question-level candidate-oracle EM rate:
whether any valid candidate has EM=1. The baseline comparator is the unchanged
baseline candidate on the same eligible questions. Also report candidate
uniqueness, parse failures, F1, verifier choice, order agreement, and timing.

The 50-question validation[850:900] screen checks contracts, slice
disjointness, baseline equivalence, candidate diversity, gold isolation, order
balance, privacy, and runtime. A normal null screen is inconclusive.

Only the 200-question validation[900:1100] formal run can decide the path:

- candidate-oracle EM uplift >= 0.05 absolute and paired-bootstrap 95% lower
  bound > 0;
- verifier EM uplift >= 0.03 absolute, paired-bootstrap lower bound > 0;
- verifier recovers >= 60% of the oracle EM gain;
- verifier F1 loss is no more than 0.01.

If oracle coverage passes but verifier fails, stop before any production change and
keep only a separate ranking research question. If oracle coverage fails, stop
the candidate path. A positive formal verifier result authorizes only the
disjoint replication slice validation[1100:1300]; it never authorizes a
selector or Generator mutation by itself.

## Budget and provenance

Worst-case screen budget is 300 generation-equivalent calls per 50 questions:
one baseline, one conditional copy control, two additional candidates, and two
verifier order passes per eligible question. The formal budget is 1,200 calls
for 200 questions under the same frozen contract.

Every run must record:

- git commit and status;
- config SHA-256 and plugin-tree hash;
- resolved model ID, revision, path, and config.json hash;
- retrieval and QORE settings;
- candidate/verifier profile IDs and ordering seed;
- per-arm counts, scalar metrics, timing, and output paths.

## Expected failure signatures

Stop or revise on any of the following:

- candidate prompts contain gold or evaluator values;
- candidate modes collapse to identical outputs on synthetic diversity fixtures;
- verifier choice changes under the fixed order permutation;
- candidate oracle uses verifier output or post-hoc labels before inference ends;
- compact artifacts contain questions, passages, candidates, predictions, prompts, or
  support reasons;
- formal oracle or verifier gates fail.

## Collision and claim boundary

Candidate generation, self-consistency, grounded verification, and answer
reranking are established neighboring techniques. This plan does not claim a new
algorithm. Its only research value is the causal decomposition of the current
QORE failure mode. A positive diagnostic result is evidence for where to work,
not evidence of a production improvement.
