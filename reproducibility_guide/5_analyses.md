# Manuscript analyses

The numbered analysis directories encode the intended order. Once all shared
inputs exist, independent branches can run concurrently.

## Select the metric set

All analysis launchers use the same `ANALYSIS_SET` environment variable:

| Value | Purpose |
| --- | --- |
| `primary` | Main analyses and main-text artifacts; this is the default |
| `full` | Process all metrics and write both primary and supplemental result views |

The analytic replicator normally runs the default `primary` workflow. No
additional flag is required. For an explicit primary-only session, run:

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

## Minimal main-text workflow

For Figures 1–5 and Table 4, the required computational workflow is:

1. Run all three launchers in `processing/03_registration_and_warping/` in
   dependency order: T1w registration first, then the DKT and bundle warps in
   parallel.
2. Run all three launchers in `analysis/00_prepare_inputs/`: the ribbon masks,
   DKT parcel statistics, and bundle statistics. The DKT and bundle summaries
   must wait for their corresponding warp jobs.
3. Run these five main-analysis launchers:

   - `analysis/02_gm_wm_differentiation/01_mni_effect_sizes/submit.sbatch`
   - `analysis/03_correlations/01_mni_voxelwise/submit.sbatch`
   - `analysis/04_icc/01_parcel_bundle/submit.sbatch`
   - `analysis/04_icc/02_mni_voxelwise/submit.sbatch`
   - `analysis/05_discriminability/02_parcel_bundle/submit.sbatch`

4. Run `python analysis/01_build_missingness_list.py` for Figure 2.

The regional-correlation launcher and MNI voxelwise-discriminability launcher
are not required by the current main-text figure or table generators. Regional
correlations are required for Figures S3–S6. Voxelwise discriminability is an
available analysis output, but no current manuscript artifact consumes it.

## GM/WM differentiation

The default job produces the primary results for Figure 3:

```bash
effect_size_job=$(sbatch --parsable \
  --dependency="afterok:${ribbon_job}" \
  analysis/02_gm_wm_differentiation/01_mni_effect_sizes/submit.sbatch)
```

## Correlations

Voxelwise correlations use MNI-space scalar maps and ribbon masks:

```bash
mni_correlation_job=$(sbatch --parsable --dependency="afterok:${ribbon_job}" \
  analysis/03_correlations/01_mni_voxelwise/submit.sbatch)
```

Parcel/bundle correlations use both regional summary branches:

```bash
regional_correlation_job=$(sbatch --parsable \
  --dependency="afterok:${dkt_stats_job}:${bundle_stats_job}" \
  analysis/03_correlations/02_parcel_bundle/submit.sbatch)
```

Primary MNI voxelwise results support Figure 4. Full voxelwise and regional
results support Figures S3–S6.

## Intraclass correlation

Regional ICC uses the DKT and bundle tables:

```bash
regional_icc_job=$(sbatch --parsable \
  --dependency="afterok:${dkt_stats_job}:${bundle_stats_job}" \
  analysis/04_icc/01_parcel_bundle/submit.sbatch)
```

Voxelwise ICC may run independently once the configured source derivatives are
available:

```bash
mni_icc_job=$(sbatch --parsable \
  analysis/04_icc/02_mni_voxelwise/submit.sbatch)
```

The regional ICC output includes `within_subject_sd` and
`between_subject_sd`. Figure S7 calculates their ratio directly from these
tables. Primary ICC results support Figure 5, while full ICC results support
Figures S8–S11.

## Discriminability

```bash
mni_discriminability_job=$(sbatch --parsable \
  analysis/05_discriminability/01_mni_voxelwise/submit.sbatch)

regional_discriminability_job=$(sbatch --parsable \
  --dependency="afterok:${dkt_stats_job}:${bundle_stats_job}" \
  analysis/05_discriminability/02_parcel_bundle/submit.sbatch)
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

## Full workflow with supplemental metrics

The minimal workflow above is sufficient for the main-text artifacts. In a
complete reproduction, add the expanded supplemental results by submitting
the relevant launchers with `ANALYSIS_SET=full`:

```bash
sbatch --dependency="afterok:${ribbon_job}" \
  --export=ALL,ANALYSIS_SET=full \
  analysis/02_gm_wm_differentiation/01_mni_effect_sizes/submit.sbatch

sbatch --dependency="afterok:${ribbon_job}" \
  --export=ALL,ANALYSIS_SET=full \
  analysis/03_correlations/01_mni_voxelwise/submit.sbatch

sbatch --dependency="afterok:${dkt_stats_job}:${bundle_stats_job}" \
  --export=ALL,ANALYSIS_SET=full \
  analysis/03_correlations/02_parcel_bundle/submit.sbatch

sbatch --export=ALL,ANALYSIS_SET=full \
  analysis/04_icc/02_mni_voxelwise/submit.sbatch

sbatch --dependency="afterok:${dkt_stats_job}:${bundle_stats_job}" \
  --export=ALL,ANALYSIS_SET=full \
  analysis/04_icc/01_parcel_bundle/submit.sbatch

sbatch --dependency="afterok:${dkt_stats_job}:${bundle_stats_job}" \
  --export=ALL,ANALYSIS_SET=full \
  analysis/05_discriminability/02_parcel_bundle/submit.sbatch
```

These jobs provide Figure S2, Figures S3–S6, Figures S8–S11, and Figure S12.
Figure S7 uses primary regional ICC results. Full voxelwise discriminability
can also be requested with the same flag, although no supplemental plotting
script requires it.

Because `full` also writes the primary result view, these commands may be used
from the outset instead of first running the primary analysis commands.

## Monitor jobs

```bash
squeue -u "${USER}"
```

Job logs are written below the profile's `logs_dir`. A failed array task should
be diagnosed and rerun before figures are generated; otherwise plotting scripts
may skip missing inputs or stop in strict mode.
