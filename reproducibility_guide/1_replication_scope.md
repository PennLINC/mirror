# Replication scope and data boundary

## Reusable inputs

**Lab replication begins after all metric processing is complete**, at
[registration and warping](5_registration_and_warping.md). After
[setup](2_setup.md), the lab replicator can go directly to that chapter.

The guide also documents [curation](3_curation.md) and
[derivatives creation](4_derivatives_creation.md) for readers regenerating
the dataset. Their outputs are precomputed inputs to lab replication:

- `smriprep`
- `qsiprep`
- `qsirecon`
- `pymp2rage`
- `ihmt`
- `mese`
- `megre`
- `qsm`
- `t1wt2w_ratio`
- `q_ratio`
- `g_ratio`

These directories normally live immediately below `source_derivatives_dir`.
The scalar QSIRecon subdirectories are declared in the selected profile.
Bundle warping also requires `qsirecon/derivatives/qsirecon-MSMTAutoTrack`.

## Steps not repeated

The full guide describes the following upstream steps, but **the lab
replicator does not need to run them because they are presumed already
complete**:

- `curation/`: DICOM-to-BIDS conversion and dataset preparation, described in
  [data curation](3_curation.md).
- `processing/01_smri_dmri/`: sMRIPrep, QSIPrep, and QSIRecon, described in
  [derivatives creation](4_derivatives_creation.md).
- `processing/02_other_metric_processing/`: source scalar-map generation,
  also described in [derivatives creation](4_derivatives_creation.md).

The released BIDS data and precomputed derivatives can be downloaded from
[MIRROR on OpenNeuro (ds008825)](https://openneuro.org/datasets/ds008825)
instead of regenerated. Point the replication profile's `bids_dir` and
source-derivative paths to the downloaded data, as described in
[setup](2_setup.md).

These upstream chapters document how the data were created for readers who
want to regenerate them. Both runnable replication pages—the minimal
main-text workflow and the full analysis workflow with supplemental
results—start from the precomputed derivatives. The full analysis workflow
does not require repeating these upstream steps.

```{warning}
Do not use the precomputed `MTssat` or `MTssat-B1c` maps. The modified
`ihmt_proc` version used by the source-scalar workflow incorrectly calculated
both from the dual-frequency saturation estimate. They are excluded from the
analysis metric registry and therefore from both `primary` and `full` analysis
modes. The retained files are processing provenance only; see
[MIRROR issue 22](https://github.com/PennLINC/mirror/issues/22).
```

## Choose a runnable workflow

For commands collected in execution order, including cloning, environment
setup, and profile selection, use either of these standalone pages:

- [Minimal main-text workflow](10_minimal_main_text_workflow.md): primary
  analyses for Figures 1–5 and Table 4.
- [Full workflow](11_full_workflow.md): primary and supplemental analyses,
  figures, and tables, plus optional voxelwise discriminability.

Both pages start from precomputed source derivatives and include explicit
wait points between jobs. The chapters below explain each stage in detail.

## Expected project layout

One possible cluster layout is:

```text
<project_root>/
├── apptainer/
├── code_replication/             # this repository; any name/location is valid
│   └── logs/                     # ignored Slurm output and job records
├── derivatives/
│   ├── smriprep/                 # reusable inputs
│   ├── qsiprep/
│   ├── qsirecon/
│   ├── pymp2rage/
│   ├── ihmt/
│   ├── mese/
│   ├── megre/
│   ├── qsm/
│   ├── t1wt2w_ratio/
│   ├── q_ratio/
│   ├── g_ratio/
│   └── replication/              # new outputs
├── dset/                         # raw BIDS dataset
└── work/
```

The source and output trees may live elsewhere, including on different
filesystems, as long as the profile contains their absolute paths.

## Run output tree

A run including optional analyses and QC will resemble:

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

The minimal main-text workflow does not create the optional regional-correlation
or voxelwise-discriminability directories. QC directories appear when their
report jobs are run. Some filenames include method or analysis-set qualifiers.
The profile, not the checkout location, determines the output root.
