# B0-orientation sanity check

Tests whether DTI metrics vary with fiber orientation relative to B0, using the natural
head-pose differences between the two NIBS sessions. The design and rationale are in
[b0_orientation_fa_tractometry_analysis_plan.md](b0_orientation_fa_tractometry_analysis_plan.md).

## Environment

```bash
wsl -e bash -lc "micromamba create -y -f fun-analyses/environment_b0_orientation.yml"
```

## Pipeline

Outputs go to `derivatives.b0_orientation` in `configuration/paths_<config>.yml`. On the PC
this is `Documents/datasets/nibs-b0-orientation`.

| Step | Script | Output |
|------|--------|--------|
| 1 | `extract_b0_pose.py` | `b0_pose*.tsv`: B0 direction in ACPC space per session (DWI registration + anatomical chain; MEGRE diagnostic) |
| 2 | `b0_tractometry.py --n-jobs 8` | `nodes/sub-*_ses-*_nodes.parquet`: 50-node profiles per AutoTrack bundle, with orientation moments (scatter tensor `M_*`, fourth order `Q_*`); `xfm/`: T1w→ACPC transforms for R2* |
| 3 | `run_b0_analysis.py` | `results/*.tsv`, `b0_orientation_report.html` |

```bash
wsl -e bash -lc "cd /mnt/c/Users/tsalo/Documents/linc/nibs && micromamba run -n b0orient python fun-analyses/run_b0_analysis.py"
```

`b0_tractometry.py` skips sessions that already have outputs unless `--overwrite` is given.
`run_b0_analysis.py --quick` uses few resamples, for testing.

The primary angular function is sin⁴θ, and estimates are compared with the tract-segment
results of Kleban, Jones & Tax (2023, doi:10.1162/imag_a_00012); see plan Sections 1.1 and 7.
`REFERENCE_EFFECTS` in `run_b0_analysis.py` holds those values.

Outputs with a `_v1` suffix in the output directory are from the first run (primary function
sin²θ, node tables without fourth-order moments).

## Tests

```bash
wsl -e bash -lc "cd /mnt/c/Users/tsalo/Documents/linc/nibs/fun-analyses && micromamba run -n b0orient python -m pytest -q tests"
```
