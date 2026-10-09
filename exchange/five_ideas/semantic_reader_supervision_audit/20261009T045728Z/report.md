# Train-only supervision audit

32 cases x 50 passages; replay only; no training/Generator/evaluation.

- frozen_reader_topk: 87 weak-positive selected; 0 changed sets.
- quantum_semantic: 87 weak-positive selected; 0 changed sets.
- classical_semantic: 87 weak-positive selected; 0 changed sets.
- quantum_scalar_control: 89 weak-positive selected; 4 changed sets.
- monotonic_compression_control: 87 weak-positive selected; 0 changed sets.

**Weak counts are not support accuracy.** Blinded support review is pending. Old heads trained on every sampled case.
