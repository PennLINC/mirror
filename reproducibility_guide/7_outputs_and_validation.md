# Outputs and validation

## Run output tree

A completed replication will resemble:

```text
<output_derivatives_dir>/
├── t1w_registration/
├── warped_bundles/
├── mni_ribbon_masks/
├── DKTatlas_myelin_stats/
├── bundle_myelin_stats/
├── missingness/
├── mni_gm_wm_effect_sizes/
├── mni_voxelwise_correlations/
├── parcel_bundle_correlations/
├── mni_voxelwise_icc/
├── parcel_bundle_icc/
├── mni_voxelwise_discriminability/
├── parcel_bundle_discriminability/
├── quality_control/
├── qc_report/
└── figures/
```

Some directory names may include additional method or analysis-set qualifiers.
The profile, not the checkout location, determines the output root.

## Confirm output isolation

Before and after a test job, print the selected paths:

```bash
python configuration/resolve_paths.py
```

No post-scalar command in this guide should use a source-derivative directory
as its output unless the user explicitly overrides a script argument or points
`output_derivatives_dir` there.

## Recommended one-participant smoke test

Before submitting all array tasks:

1. edit the replication profile and select a fresh output directory;
2. temporarily change the relevant SBATCH array range to one participant in an
   untracked copy of the launcher, or submit a single array index with
   `sbatch --array=1`;
3. run T1w registration, both warp branches, and their dependent regional
   summaries;
4. inspect the generated paths and QC reportlets; and
5. confirm the original manuscript derivative tree is unchanged.

For example:

```bash
sbatch --array=1 \
  processing/03_registration_and_warping/01_submit_t1w_registration.sbatch
```

## Compare with manuscript artifacts

Comparison should focus on:

- participant/session inclusion counts;
- missing-input and QC diagnostics;
- metric inclusion tables;
- numerical summary tables before rendered figures; and
- visible agreement of figure ordering, axes, and values.

Exact raster pixels can vary with Matplotlib, font, and system-library versions.
Numerical tables are the preferred basis for confirming analytic replication.

## Software record

Record at minimum:

```bash
git rev-parse HEAD
python --version
micromamba list --explicit > "${MIRROR_OUTPUT_DERIVATIVES}/environment-explicit.txt"
```

Also retain the replication profile and Slurm job IDs with the output dataset,
excluding secrets or access tokens.
