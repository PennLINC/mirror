# Replication scope and data boundary

## Reusable inputs

Replication starts after preprocessing and source-scalar generation. The
following derivative datasets are treated as inputs:

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
The exact QSIRecon subdirectories are declared in the selected profile.

## Steps not repeated

The replication described here does not run:

- `curation/`, which documents DICOM-to-BIDS curation;
- `processing/01_smri_dmri/`, which runs sMRIPrep, QSIPrep, and QSIRecon; or
- `processing/02_other_metric_processing/`, which generates source scalar
  derivatives.

Those workflows remain in the repository for provenance and for users who need
to regenerate the released source data.

```{warning}
Do not use the precomputed `MTssat` or `MTssat-B1c` maps. The modified
`ihmt_proc` version used by the source-scalar workflow incorrectly calculated
both from the dual-frequency saturation estimate. They are excluded from the
analysis metric registry and therefore from both `primary` and `full` analysis
modes. The retained files are processing provenance only; see
[MIRROR issue 22](https://github.com/PennLINC/mirror/issues/22).
```

## First replicated step

Begin with:

```text
processing/03_registration_and_warping/
```

All later computational outputs are written beneath the configured
`output_derivatives_dir`. Generated figures and tables default to its
`figures/` subdirectory. Logs use the checkout's ignored `logs/` directory;
temporary files use the configured `work_dir`.

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
