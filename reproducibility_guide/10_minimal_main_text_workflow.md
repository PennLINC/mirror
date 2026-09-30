# Minimal main-text workflow

This page collects the setup and commands needed for **Figures 1–5 and
Table 4**, using `ANALYSIS_SET=primary`. It assumes the reusable source
metrics already exist: lab replication begins after metric processing.
For supplemental artifacts, use the [full workflow](11_full_workflow.md),
which also produces the main-text results.

## 1. Clone, create the environment, and select the profile

These commands use the CUBIC paths. The lab replicator can use the included
profile without modification; for another system, adapt its paths as described
in [setup](2_setup.md). The source derivatives and required external software
must already be available.

Run the clone and environment-creation commands once. If this checkout and
`processing` environment already exist, skip those two commands.

```bash
git clone -b code_reorg \
  git@github.com:PennLINC/mirror.git \
  /cbica/projects/nibs/code_replication
cd /cbica/projects/nibs/code_replication
conda env create -f environment_processing.yml --channel-priority=flexible
```

Run this block at the start of each working shell:

```bash
cd /cbica/projects/nibs/code_replication
conda activate processing
export MIRROR_CONFIG="${PWD}/configuration/profiles/replication.example.yml"
export ANALYSIS_SET=primary
command -v python
python configuration/resolve_paths.py
bash configuration/create_log_directories.sh
```

Confirm that the Python executable belongs to `processing`, the source paths
point to the precomputed derivatives, and the output and work paths point to
this replication. The default output root is
`/cbica/projects/nibs/derivatives/replication`; use a new output directory for
an independent run. Reusing that directory may replace its earlier results.

```{important}
Submit every `sbatch` command from the repository root in this configured
shell. **Run the blocks in order and stop at each wait point.** Submission
returns immediately; downstream jobs must wait until all required upstream
array tasks finish successfully.
```

At each wait point, inspect the queue and the job IDs printed by `sbatch`:

```bash
squeue -u "${USER}"
# Replace 123456 with the job ID to inspect, including its array tasks.
sacct -j 123456 --format=JobID,State,ExitCode
```

Check for successful completion and inspect failures in `<checkout>/logs/`
before continuing. Leaving the queue alone does not establish success.

## 2. Registration and warping

Create the transforms first:

```bash
sbatch processing/03_registration_and_warping/01_submit_t1w_registration.sbatch
```

**Wait for all registration tasks to succeed**, then submit both warp jobs:

```bash
sbatch processing/03_registration_and_warping/02_submit_dkt_atlas_warping.sbatch

sbatch processing/03_registration_and_warping/03_submit_bundle_warping.sbatch
```

## 3. Shared analysis inputs

**Wait for both warp arrays to succeed.** These jobs create MNI ribbon masks,
DKT parcel summaries, bundle summaries, and the acquisition-availability table:

```bash
sbatch analysis/00_prepare_inputs/01_mni_ribbon_masks/submit.sbatch

sbatch analysis/00_prepare_inputs/02_dkt_parcel_stats/submit.sbatch

sbatch analysis/00_prepare_inputs/03_bundle_myelin_stats/submit.sbatch

python analysis/01_build_missingness_list.py
```

## 4. Analyses

**Wait for all shared-input jobs to succeed**, then submit these five
independent analyses together. With `ANALYSIS_SET=primary` selected above,
they compute the main-text GM/WM effect sizes, voxelwise correlations,
regional and voxelwise ICC, and regional discriminability. See
[manuscript analyses](7_analyses.md) for their purposes and output paths.

```bash
sbatch analysis/02_gm_wm_differentiation/01_mni_effect_sizes/submit.sbatch

sbatch analysis/03_correlations/01_mni_voxelwise/submit.sbatch

sbatch analysis/04_icc/01_parcel_bundle/submit.sbatch

sbatch analysis/04_icc/02_mni_voxelwise/submit.sbatch

sbatch analysis/05_discriminability/02_parcel_bundle/submit.sbatch
```

These are the five analysis jobs required by the current main-text artifacts.
The regional-correlation job is supplemental, and no current figure or table
requires the MNI voxelwise-discriminability job.

Wait for every required job and array task to finish successfully.

## 5. Quality control

The report purposes and output paths are described in
[derivatives creation](4_derivatives_creation.md#quality-control-reports).
The consolidated report can now include the completed analysis outputs.

```bash
sbatch processing/04_quality_control/01_coregistration_reports/submit.sbatch
sbatch processing/04_quality_control/02_scalar_reports/submit.sbatch
sbatch processing/04_quality_control/03_analysis_qc/submit.sbatch
```

Wait for the report jobs to finish and review the reports before accepting
the results.

## 6. Main figures and table

These commands render Figures 1–5 and Table 4 beneath
`<output_derivatives_dir>/figures/`. See
[figures and tables](8_figures_and_tables.md) for the input-to-artifact mapping.

```bash
python figures/figure_01_primary_maps/plot_figure_1_primary_maps.py
python figures/figure_02_missingness/plot_figure_2_missingness.py
python figures/figure_03_gm_wm_effect_sizes/plot_figure_3_gm_wm_effect_sizes.py
python figures/figure_04_correlations/plot_figure_4_correlations.py
python figures/figure_05_icc/plot_figure_5_icc.py

python figures/table_04_discriminability/make_table_4_discriminability.py
```
