# Close the current quantum Reader rotation configuration; repair objective validity before another screen

- Date: 2026-10-16
- Scope: rag-selector
- Status: active
- Supersedes: None; records the outcome of 2026-10-13-quantum-reader-rotation-screen.md
- Superseded By: None
- Gate: source-complete loss dispatch and scale/membership counterexamples pass before a new paired experiment; utility requires positive-consensus paired lower CI >0 and nondecreasing direct support; attribution separately requires gain over matched classical control
- Authorization: analyze and record completed artifacts only; no new fitting, dataset download, candidate implementation, or rerun in this closeout

## Decision

Close this 200-parameter rotation/new-loss configuration after the fixed100 L0 screen. Do not promote to full-data execution, L1/L2, or a quantum-advantage claim. Do not close the quantum Reader family solely on this result.

The next recommended research step is a train-cache scale-invariance and actual Top5-membership objective audit, retaining the quantum Reader mainline and a matched classical control. Restore exact executed-source provenance and independent loss dispatch before another experiment. Do not blindly increase epochs, residual freedom, circuit depth or data.

## Evidence

Run `20261015T022004545902Z`: Reader/quantum/classical positive totals203/147/170; direct138/90/112; fixedSilver355/175/200. Quantum-minus-Reader positive mean−0.56, paired95%CI[−0.80,−0.32]; quantum-minus-classical−0.23,[−0.40,−0.07]. Primary and attribution gates failed. Six source artifact hashes/receipts match. All954 rankings and600 posthoc rows match;795 cached-feature head replays match, no Top5 order difference, maxabs9.06e-6. Only selected4651/9318 arrays received finite numerical checks; full source NPZ SHA was checked remotely.

Training illustrates why improved pairs are insufficient: quantum119/228 inversions but only22/135 eligible candidates enter Top5 and104/183 protected candidates are lost; classical116/228,28 rescues,78 protected losses. Those183 include partial/conservative protected items, not183 direct positives. Quantum changes97 evaluation sets, adds57/removes113 useful passages. Score std mean shrinks4.159→0.456 in training and3.616→0.612 in evaluation. A fixed positive affine compression to10% leaves every ranking unchanged yet reduces mean objective3.979→1.065; quantum final objective1.177. This supports an objective shortcut, without proving it is the only cause of harmful ranking changes.

Old/new named loss arms are exact duplicates within each head family. Quantum old-loss witnesses match new loss6/6, with late witnesses incompatible with old loss; the head/loss ablation is invalid. Exact execution runner SHA differs from reportedGit commit and is not supplied; config execution SHA can be reconstructed with the full training-bundle hash. Repository delivery has loss-dispatch, target-wrapper and truncated-hash defects. Given frozen features and final checkpoints, primary output scores and metrics remain independently checkable; full optimizer/source replay and causal loss comparison do not.

## Consequences

- Preserve immutable artifacts and all results; no posthoc label/panel changes or gate relaxation.
- Distinguish current configuration failure, invalid cross-loss ablation, and source-completeness limitations. Do not conflate these into a universal mechanism verdict.
- Before a new screen require an explicit arm→head/objective/config table, counterexamples that distinguish loss variants, target schema checks, and archived executed source bytes/patches matching provenance hashes. Synthetic replay alone did not catch the contract defect.
- Reject any next objective that can materially reward positive affine score compression without improving ordering or preserving actual Top5 support. A future loss change is an enabling control, not itself evidence of quantum novelty.
- Log: [owner-readable analysis](../../../../docs/rag-research-log/20261015T-022004Z-quantum-reader-rotation-screen-result.md). Receipt: [compact audit](../refs/quantum_reader_rotation_screen_20261015_receipt.json).
- Versioned explanation/portal association remains a separate pending documentation closeout; do not imply it was published.
