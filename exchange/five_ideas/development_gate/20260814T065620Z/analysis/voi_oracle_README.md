# Adaptive VoI Oracle Diagnostic

This is a passage-free oracle replay over the recorded qore_as_control candidate pools. `candidate_flags.is_gold` supplies the diagnostic oracle label; it is not available to a deployed selector. The tau sweep adds a bounded uncertainty transform `4*a*(1-a)` to the existing quality signal and re-solves the exact recorded QUBO. It measures headroom and plausibility only; it is not held-out evidence and does not estimate new generation F1.

A VoI implementation should proceed only if the oracle gap is material and the uncertainty transform closes a reproducible fraction of it without excessive selection cost.
