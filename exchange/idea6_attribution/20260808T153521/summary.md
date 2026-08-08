# Idea 6 Attribution Matrix

- Run: 20260808T153521
- Samples: 3610
- Seed: 42

## Overall Metrics

| Config | Recall@5 | F1 | EM | Redundancy | Selection ms |
|---|---:|---:|---:|---:|---:|
| qore_dpr | 0.3407 | 0.4344 | 0.2648 | 0.8110 | 11.2503 |
| qore_as_control | 0.4743 | 0.4714 | 0.2956 | 0.7846 | 11.8665 |
| qore_as_idea6 | 0.4802 | 0.4737 | 0.2970 | 0.7896 | 1302.7079 |
| topk_as | 0.5167 | 0.4729 | 0.2936 | 0.8230 | 0.1047 |
| mmr_as | 0.4933 | 0.4715 | 0.2936 | 0.7962 | 1.2517 |

## Paired Effects

Positive delta means the right-hand configuration is higher.

| Comparison | Metric | Mean delta | Wins | Losses | Ties | Bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---:|
| qore_dpr -> qore_as_control | f1 | +0.0371 | 717 | 497 | 2396 | [+0.0253, +0.0480] |
| qore_dpr -> qore_as_control | recall | +0.1336 | 1413 | 458 | 1016 | [+0.1210, +0.1470] |
| qore_dpr -> qore_as_control | em | +0.0307 | 301 | 190 | 3119 | [+0.0183, +0.0418] |
| qore_dpr -> qore_as_control | redundancy | -0.0264 | 1006 | 2603 | 1 | [-0.0282, -0.0246] |
| qore_as_control -> qore_as_idea6 | f1 | +0.0023 | 121 | 109 | 3380 | [-0.0016, +0.0063] |
| qore_as_control -> qore_as_idea6 | recall | +0.0059 | 173 | 100 | 2614 | [+0.0023, +0.0095] |
| qore_as_control -> qore_as_idea6 | em | +0.0014 | 44 | 39 | 3527 | [-0.0036, +0.0064] |
| qore_as_control -> qore_as_idea6 | redundancy | +0.0050 | 1132 | 302 | 2176 | [+0.0045, +0.0055] |

## Interpretation

- qore_dpr -> qore_as_control estimates the Answer Scorer effect.
- qore_as_control -> qore_as_idea6 is the matched complementarity effect.
- Compare qore_as_idea6 with topk_as and mmr_as under the same Answer Scorer.
- Use question-level bootstrap; the seed is not an independent replicate.
