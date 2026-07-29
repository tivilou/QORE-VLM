# Phase 2 Script Changelog

All notable changes to the P2 experiment scripts.

---

## [2026-07-29] - Documentation Reorganization

### Changed
- Moved script documentation from `exchange/` to `scripts/collab/p2_solver_idea6/`
- Organized docs: README (overview), SETUP (guide), CHANGELOG (history)
- Detailed fix reports moved to `docs/` subdirectory

### Rationale
- Documentation should live with the code, not in experiment results directory
- Makes it easier for collaborators to find setup instructions
- Clearer separation: scripts vs. experiment results

---

## [2026-07-28] - Corpus Mode Switch & Aligned Fix

### Changed
- **Switched to `wiki_dpr` corpus mode** (from `aligned`)
  - More realistic: uses full 21M Wikipedia corpus
  - Already verified working on collaborator's machine
  - Better for final parameter tuning

### Fixed
- **Aligned mode freeze issue**:
  - Fixed default config: `psgs_w100.nq.exact` → `psgs_w100.nq.compressed`
  - Added progress logging (every 1000 items or 2 seconds)
  - Users won't think it's frozen anymore

### Added
- Wiki_dpr mode configuration variables:
  - `WIKI_DPR_CONFIG="psgs_w100.nq.compressed"`
  - `WIKI_DPR_CACHE_DIR="/root/.cache/huggingface/datasets"`
- Conditional parameter passing for wiki_dpr mode

### Technical Details
- Root cause: aligned mode used non-existent config + slow streaming (108k items) + no progress logs
- See `docs/ALIGNED_FIX_SUMMARY.md` for full diagnosis

---

## [2026-07-28] - Parameter Support for Idea 6

### Added
- **New CLI parameters**:
  - `--delta`: Complementarity weight (default 0.0)
  - `--complementarity_method`: How to compute complementarity (choices: dpr, None)

### Fixed
- **PYTHONPATH issue**: Added `export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"`
- **REPO_ROOT path**: Fixed relative path resolution (`../..` → `../../..`)

### Changed
- Updated `eval_rag_refactored.py` to support complementarity parameters
- Updated `select_passages()` call with 5 new parameters:
  - `delta`
  - `complementarity_method`
  - `answer_scorer`
  - `passage_texts`
  - `question`

### Technical Details
- Enables Phase 2 experiments: solver fix + idea 6 parameter grid
- Complementarity matrix rewards selecting diverse passage pairs
- DPR answer scorer required when `complementarity_method='dpr'`
- See `docs/P2_FIX_REPORT.md` for detailed technical report

### Verification
- 7/7 validation tests passed
- Cross-machine compatibility verified
- PYTHONPATH works in all execution scenarios

---

## [2026-07-28] - Initial Script Creation

### Added
- `run_p2_experiments.sh`: Automated P2 experiment runner
  - 10 experiment configurations (1 baseline + 9 idea6 grid)
  - Gamma ∈ {0.3, 0.5, 0.7}
  - Delta ∈ {0.1, 0.3, 0.5}
  - 200 samples per experiment
- `collect_p2_results.py`: Result collection and README generation
- Automatic timestamp handling (Beijing timezone)
- Output organized under `exchange/p2_solver_idea6/`

### Features
- Parameter validation checks before execution
- Automatic result aggregation
- Git commit message suggestion
- Progress tracking

---

## Usage Notes

### Running Experiments
```bash
cd /path/to/QORE-VLM
bash scripts/collab/p2_solver_idea6/run_p2_experiments.sh
```

### After Updates
```bash
git pull origin main  # Always pull latest changes first
```

### Checking Compatibility
```bash
# Verify parameters exist
python -m scripts.rag.eval_rag_refactored --help | grep -E "delta|complementarity"

# Test with 1 sample
bash scripts/collab/p2_solver_idea6/run_p2_experiments.sh  # Edit MAX_SAMPLES=1 first
```

---

## Related Documentation

- `README.md` - Script overview and quick start
- `SETUP.md` - Detailed setup guide for collaborators
- `docs/P2_FIX_REPORT.md` - Parameter fix technical report
- `docs/ALIGNED_FIX_SUMMARY.md` - Aligned mode fix explanation
- `../../docs/rag/corpus_modes.md` - Corpus mode technical guide
- `../../docs/rag/troubleshooting.md` - Common issues and solutions

---

## Commit References

- `586a1bd` - docs: reorganize documentation by audience and purpose
- `1ab79b7` - fix(rag): fix aligned mode freeze and switch P2 to wiki_dpr
- `80ad6c7` - feat(rag): add complementarity parameters for idea 6 experiments
- `79cebc1` - refactor(collab): reorganize scripts by phase with workflow docs
