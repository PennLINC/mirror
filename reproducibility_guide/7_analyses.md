# Manuscript analyses

The numbered analysis directories encode the intended order. Once all shared
inputs exist, independent branches can run concurrently. **Wait for the
required preparation jobs to finish successfully before submitting analyses.**
All output directories below are relative to `output_derivatives_dir`.

## Select the metric set

All analysis launchers use the same `ANALYSIS_SET` environment variable:

| Value | Purpose |
| --- | --- |
| `primary` | Main analyses and main-text artifacts; this is the default |
| `full` | Process all metrics and write both primary and supplemental result views |

The analytic replicator normally runs the default `primary` workflow. No
additional flag is required. You may also explicitly specify:

```bash
export ANALYSIS_SET=primary
```

All subsequent `sbatch` submissions inherit that value. To run a launcher with
the supplemental metric set, add `--export=ALL,ANALYSIS_SET=full` to its
`sbatch` command. Output tables identify the selected analysis set, and most
set-specific filenames include `primary` or `full`.

```{important}
`full` includes every primary metric. It also refreshes the primary result view
before writing the expanded supplemental view. Therefore, running `full` after
`primary` in the same output root is safe: primary summaries may be replaced by
equivalent recomputed versions, while supplemental results are added.
```

## GM/WM differentiation

This job measures how strongly each metric separates gray and white matter
within the MNI ribbon masks. It writes session, participant, and group-summary
effect-size TSVs, with diagnostic and metric-inclusion tables, beneath
`mni_gm_wm_effect_sizes/`. The primary summaries support Figure 3:

```bash
sbatch analysis/02_gm_wm_differentiation/01_mni_effect_sizes/submit.sbatch
```

## Correlations

Correlations quantify spatial correspondence between metrics. The voxelwise
job uses MNI-space scalar maps and ribbon masks, writing correlation and
Fisher-z matrices plus coverage and participant-count TSVs beneath
`mni_voxelwise_correlations/`:

```bash
sbatch analysis/03_correlations/01_mni_voxelwise/submit.sbatch
```

The parcel/bundle job correlates regional metric profiles using both summary
branches. It writes correlation matrices, feature counts, participant counts,
and metric-inclusion TSVs beneath `parcel_bundle_correlations/`:

```bash
sbatch analysis/03_correlations/02_parcel_bundle/submit.sbatch
```

Primary MNI voxelwise results support Figure 4. Full voxelwise and regional
results support Figures S3–S6.

## Intraclass correlation

Intraclass correlation (ICC) measures test–retest reliability across sessions.
The regional job uses the DKT and bundle tables and writes per-region CSV
results beneath `parcel_bundle_icc/`, including ICC and variance components:

```bash
sbatch analysis/04_icc/01_parcel_bundle/submit.sbatch
```

Voxelwise ICC may run independently once the configured source derivatives are
available:

```bash
sbatch analysis/04_icc/02_mni_voxelwise/submit.sbatch
```

The voxelwise job writes NIfTI ICC and supporting statistical maps, summary
and diagnostic TSVs, and metadata beneath `mni_voxelwise_icc/`.

The regional ICC output includes `within_subject_sd` and
`between_subject_sd`. Figure S7 calculates their ratio directly from these
tables. Primary ICC results support Figure 5, while full ICC results support
Figures S8–S11.

## Discriminability

Discriminability measures how consistently a participant’s repeated spatial
profiles resemble each other more than other participants’ profiles. The MNI
job uses reusable scalar maps and writes summary, coverage, inclusion, and
diagnostic TSVs beneath `mni_voxelwise_discriminability/`. The regional job
requires both DKT and bundle summaries and writes CSV discriminability
results beneath `parcel_bundle_discriminability/`.

```bash
sbatch analysis/05_discriminability/01_mni_voxelwise/submit.sbatch

sbatch analysis/05_discriminability/02_parcel_bundle/submit.sbatch
```

Primary regional discriminability results support Table 4; full regional
results support Figure S12. No current manuscript artifact requires MNI
voxelwise discriminability.

## Run analysis branches in parallel

No manuscript analysis depends on the output of another manuscript analysis.
Once the required shared inputs below are available, all analysis launchers may
be submitted together and Slurm may run them concurrently:

| Analysis branch | Required shared input |
| --- | --- |
| GM/WM effect sizes | MNI ribbon masks |
| MNI voxelwise correlations | MNI ribbon masks |
| Regional correlations | DKT parcel and bundle summary tables |
| Regional ICC | DKT parcel and bundle summary tables |
| MNI voxelwise ICC | Reusable source-scalar derivatives |
| MNI voxelwise discriminability | Reusable source-scalar derivatives |
| Regional discriminability | DKT parcel and bundle summary tables |

The simplest conservative schedule is to wait for the ribbon-mask, DKT-summary,
and bundle-summary jobs to finish, then submit every analysis job in this
chapter at once. The cluster scheduler controls how many actually execute at
the same time based on available resources.

## End-to-end commands

Use the [minimal main-text workflow](10_minimal_main_text_workflow.md) for
Figures 1–5 and Table 4, or the [full workflow](11_full_workflow.md) for the
main and supplemental artifacts. Each page includes setup, registration,
input preparation, analyses, QC, and figure/table commands in execution order.

## Monitor jobs

```bash
squeue -u "${USER}"
```

A job disappearing from `squeue` does not establish success. Check the job ID
printed by `sbatch` in Slurm accounting (including all array tasks):

```bash
sacct -j 123456 --format=JobID,State,ExitCode
```

Replace `123456` with that numeric ID.

Job logs are written below the profile's `logs_dir`. A failed array task should
be diagnosed and rerun before figures are generated; otherwise plotting scripts
may skip missing inputs or stop in strict mode.
