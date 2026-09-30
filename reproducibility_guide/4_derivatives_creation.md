# Derivatives creation

This chapter documents `processing/01_smri_dmri/` and
`processing/02_other_metric_processing/`: the workflows that turn curated
acquisitions into reusable anatomical, diffusion, and myelin-sensitive maps.
**All of this metric processing is complete before lab replication begins.**
The lab replicator consumes the existing outputs listed in
[replication scope](1_replication_scope.md) and starts with
[registration and warping](5_registration_and_warping.md).

## Environment and output locations

Use the `processing` environment and profile described in [setup](2_setup.md).
The creation jobs write into the configured source-derivative directories
under `source_derivatives_dir`; changing only `output_derivatives_dir` does
not redirect these upstream jobs. To regenerate derivatives separately,
configure new source-derivative paths as well.

The container download script supplies the preprocessing images:

```bash
bash processing/00_pull_apptainer_images.sh
```

The profile identifies container images, the FreeSurfer license, and the
QSM/SEPIA installation. These workflows also require the external tools used
by their scripts, including Apptainer, ANTs, MATLAB/SEPIA, and the
project-specific `ihmt_proc` API for ihMT processing.

**Submit from the repository root. Wait for every required upstream job and
array task to finish successfully before submitting a dependent stage.**
The commands below do not wait automatically.

## Anatomical and diffusion preprocessing

| Job | Purpose | Outputs under `source_derivatives_dir` |
| --- | --- | --- |
| sMRIPrep launcher | Run fMRIPrep in anatomical-only mode to prepare T₁w anatomy, tissue masks, spatial transforms, and FreeSurfer reconstructions. | `smriprep/`, including `sourcedata/freesurfer/` |
| QSIPrep launcher | Preprocess diffusion acquisitions and provide diffusion images and anatomical references in ACPC space. | `qsiprep/` |
| QSIRecon launcher | Reconstruct diffusion metrics and AutoTrack bundles from the QSIPrep outputs using `qsirecon_spec.yml`. | `qsirecon/derivatives/`, including DIPYDKI, NODDI, DSIStudio, and MSMTAutoTrack datasets |

Submit the anatomical job:

```bash
sbatch processing/01_smri_dmri/01_submit_smriprep.sbatch
```

After it succeeds, submit diffusion preprocessing:

```bash
sbatch processing/01_smri_dmri/02_submit_qsiprep.sbatch
```

After QSIPrep succeeds, submit reconstruction:

```bash
sbatch processing/01_smri_dmri/03_submit_qsirecon.sbatch
```

## Other metric processing

The table identifies each branch's prerequisite and main products. Output
folders are below `source_derivatives_dir`. The workflows also produce
supporting masks, transforms, and reportlets where applicable.

| Branch | Wait for | Purpose and main outputs |
| --- | --- | --- |
| MP2RAGE | sMRIPrep | Quantitative T₁/R₁ and supporting anatomical maps in `pymp2rage/`. |
| MESE | sMRIPrep | Transverse relaxation maps, including R₂, in `mese/`. |
| ihMT | MP2RAGE | Magnetization-transfer and ihMT metrics, including B₁-corrected variants, in `ihmt/`. |
| T₁w/T₂w ratio | MP2RAGE | Aligned anatomical ratio maps in `t1wt2w_ratio/`. |
| MEGRE | MESE | R₂* and R₂′ maps plus inputs for susceptibility processing in `megre/`. |
| QSM | MEGRE | Susceptibility and susceptibility-separation maps in `qsm/`; the launcher runs QSM processing followed by postprocessing. |
| Q-ratio | MEGRE and MP2RAGE | Q-ratio maps combining relaxation measures in `q_ratio/`. |
| g-ratio | QSIRecon and ihMT, then cohort scaling factors | Calibrated g-ratio maps in `g_ratio/`. |

After sMRIPrep, MP2RAGE and MESE may run in parallel:

```bash
sbatch processing/02_other_metric_processing/01_mp2rage/submit.sbatch
sbatch processing/02_other_metric_processing/04_mese/submit.sbatch
```

After MP2RAGE, submit the two dependent anatomical metric branches:

```bash
sbatch processing/02_other_metric_processing/02_ihmt/submit.sbatch
sbatch processing/02_other_metric_processing/03_t1wt2w_ratio/submit.sbatch
```

After MESE, submit MEGRE:

```bash
sbatch processing/02_other_metric_processing/05_megre/submit.sbatch
```

After MEGRE, submit QSM. Q-ratio can run alongside it once MP2RAGE is also
complete:

```bash
sbatch processing/02_other_metric_processing/06_qsm/submit.sbatch
sbatch processing/02_other_metric_processing/07_q_ratio/submit.sbatch
```

The g-ratio workflow has three stages. After QSIRecon and ihMT finish,
estimate participant-level scaling factors:

```bash
sbatch processing/02_other_metric_processing/08_g_ratio/01_submit_scaling_factors.sbatch
```

After **all scaling-factor array tasks** finish, aggregate the factors across
participants:

```bash
sbatch processing/02_other_metric_processing/08_g_ratio/02_aggregate_scaling_factors.sbatch
```

After aggregation succeeds, use the finalized factors to create the maps:

```bash
sbatch processing/02_other_metric_processing/08_g_ratio/03_submit_g_ratio.sbatch
```

The `MTssat` and `MTssat-B1c` outputs from the retained ihMT workflow are
excluded from analyses because of a known calculation error. See the
[scope warning](1_replication_scope.md#steps-not-repeated) for details.

## Quality-control reports

The jobs in `processing/04_quality_control/` inspect spatial alignment and
scalar-map quality. They write reports under `output_derivatives_dir`, so the
lab replicator can rerun QC on existing source derivatives without rerunning
metric creation.

After the source derivatives are available, these two independent jobs may
run together:

```bash
sbatch processing/04_quality_control/01_coregistration_reports/submit.sbatch
sbatch processing/04_quality_control/02_scalar_reports/submit.sbatch
```

| Report job | Purpose | Outputs under `output_derivatives_dir` |
| --- | --- | --- |
| Coregistration reports | Inspect alignment of acquisition and derivative images with anatomical references. | Participant HTML reports and reportlets in `quality_control/coregistration_reports/`. |
| Scalar reports | Inspect the generated myelin-sensitive maps across metrics and sessions. | Participant/session HTML reports and reportlets in `quality_control/scalar_reports/`. |
| Consolidated analysis QC | Inspect spatial analysis inputs and available results together, including registration and warped regions. | `qc_report/qc_report.pdf`, with intermediate files in `qc_report/work/`. |

**Run the consolidated report later**, after registration, warping, input
preparation, and the requested analyses have finished. It needs products
created in the following chapters:

```bash
sbatch processing/04_quality_control/03_analysis_qc/submit.sbatch
```

Review alignment overlays, scalar maps, missing-input warnings, and the
consolidated report before interpreting the replicated statistics. Job
completion alone does not establish image quality.
