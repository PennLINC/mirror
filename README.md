# MIRROR data descriptor code

Reproducible curation, processing, analysis, figure, and table code for the
MIRROR multimodal MRI data descriptor. The checkout name is not significant:
all repository assets resolve from `__file__`, while dataset paths come from a
YAML profile.

## Project layout

The expected filesystem layout is:

```text
<project_root>/
├── apptainer/              # container images
├── code/                   # this repository; contains ignored logs/
├── derivatives/            # reusable preprocessing and source scalar maps
│   └── mirror_data_descriptor/  # default manuscript-run outputs
├── dset/                   # raw BIDS dataset
└── work/
```

Set `MIRROR_CONFIG` to a profile name or YAML path. The default `hpc` profile
is cluster-oriented; `configuration/profiles/replication.example.yml` is the
starting point for another machine. A reproduction should use a distinct
`run_name` and `output_derivatives_dir`, such as
`derivatives/mirror_data_descriptor_replication`.

The reusable boundary is after source scalar generation. `smriprep`, `qsiprep`,
`qsirecon`, `pymp2rage`, `ihmt`, `mese`, `megre`, `qsm`, `t1wt2w_ratio`,
`q_ratio`, and `g_ratio` are read from `source_derivatives_dir`. Registration,
warping, regional statistics, analyses, QC products, and newly rendered
figures/tables are written beneath the run-specific `output_derivatives_dir`.
Temporary work uses `work_dir`. Scheduler output and workflow job records are
written below the checkout's ignored `logs/` directory. Run
`configuration/create_log_directories.sh` once after cloning, and submit
SBATCH launchers from the repository root.

## Repository layout

| Directory | Contents |
| --- | --- |
| `configuration/` | Machine/run profiles, authoritative `metrics.yml`, BIDS and report specifications |
| `curation/` | Numbered raw-DICOM-to-BIDS workflow |
| `processing/` | Upstream preprocessing, source-scalar generation, registration/warping, and QC |
| `analysis/` | Numbered manuscript-analysis workflow and analysis launchers |
| `figures/` | Figure/table entry points, shared plotting code, and checked manuscript artifacts |
| `data/` | Shared atlases, QC tables, and reference data |
| `utils/` | Shared path, metric, image, regional-IO, and plotting-adjacent utilities |
| `to_delete/` | Quarantined legacy code/assets pending human review |

See `processing/README.md`, `analysis/README.md`, and `figures/README.md` for
the exact run order and artifact commands.

## Environments and checks

Create the environment appropriate to the stage:

```bash
micromamba env create -f environment_curation.yml --channel-priority=flexible
micromamba env create -f environment_processing.yml --channel-priority=flexible
```

Basic repository validation:

```bash
python -m compileall -q analysis configuration curation figures processing utils
find analysis curation processing configuration -type f \
  \( -name '*.sh' -o -name '*.sbatch' \) -print0 | xargs -0 -n1 bash -n
pytest
```

## License

[MIT](LICENSE)
