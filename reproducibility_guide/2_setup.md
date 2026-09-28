# Clone, environment, and configuration

## Clone the repository

```bash
git clone -b code_reorg \
  git@github.com:PennLINC/nibs.git \
  /cbica/projects/nibs/code_replication
cd /cbica/projects/nibs/code_replication
bash configuration/create_log_directories.sh
```

The checkout does not have to be named `MIRROR` or placed directly below the
project root. The paths below reproduce the CUBIC example used for this guide;
other locations work when the profile is updated accordingly.

## Create the processing environment

The processing environment also supplies the packages used by the analyses and
figure scripts:

```bash
micromamba env create \
  -f environment_processing.yml \
  --channel-priority=flexible
micromamba activate processing
```

The curation environment is unnecessary unless the raw BIDS dataset is being
rebuilt.

## Configure the replication profile

Edit the included replication profile directly from the repository root:

```bash
nano configuration/profiles/replication.example.yml
```

JSON syntax is valid YAML and can be loaded even when PyYAML is not installed.
A cluster profile may look like:

```json
{
  "project_root": "/cbica/projects/nibs",
  "bids_dir": "dset",
  "code_dir": "auto",
  "work_dir": "work/replication",
  "source_derivatives_dir": "derivatives",
  "run_name": "replication",
  "output_derivatives_dir": "derivatives/replication",
  "logs_dir": "auto",
  "derivatives": {
    "smriprep": "smriprep",
    "qsiprep": "qsiprep",
    "qsirecon_dipydki": "qsirecon/derivatives/qsirecon-DIPYDKI",
    "qsirecon_noddi": "qsirecon/derivatives/qsirecon-NODDI",
    "qsirecon_dsistudio": "qsirecon/derivatives/qsirecon-DSIStudio",
    "pymp2rage": "pymp2rage",
    "ihmt": "ihmt",
    "mese": "mese",
    "megre": "megre",
    "qsm": "qsm",
    "t1wt2w_ratio": "t1wt2w_ratio",
    "q_ratio": "q_ratio",
    "g_ratio": "g_ratio"
  }
}
```

Retain the `apptainer`, `freesurfer`, and `software` sections from the example
when they are needed by the target cluster.

```{warning}
`configuration/profiles/replication.example.yml` is tracked by Git. After
editing it, `git status` will report a modification. Do not commit or push
cluster-specific paths unless that change is intentional. Having a locally
modified profile does not prevent job submission.
```

```{warning}
Use a new, preferably nonexistent, `output_derivatives_dir` for the replication.
Rerunning within that same directory may replace outputs from that replication.
It will not affect the manuscript run when the two profiles use different paths.
```

## Select and inspect the profile

```bash
export MIRROR_CONFIG="${PWD}/configuration/profiles/replication.example.yml"
python configuration/resolve_paths.py
```

Confirm that:

1. `MIRROR_CODE_ROOT` is the checkout you just cloned;
2. `MIRROR_SOURCE_DERIVATIVES` points to the reusable inputs; and
3. `MIRROR_OUTPUT_DERIVATIVES` and `MIRROR_WORK_DIR` point to
   replication-specific locations; and
4. `MIRROR_LOGS_DIR` is `<checkout>/logs`.

All SBATCH launchers load this same profile through
`configuration/load_profile.sh`.

The profile's relative paths are resolved against `project_root`, so this
example reads reusable inputs from `/cbica/projects/nibs/derivatives` and
writes new results to `/cbica/projects/nibs/derivatives/replication`. The
checkout itself is discovered from Git because `code_dir` is `auto`.
`logs_dir: auto` similarly follows the checkout, so renaming or moving the
clone does not require a hard-coded log path.

The SBATCH headers use relative paths such as
`logs/t1w_reg/%x-%A_%a.out`. Slurm does not create missing parent directories;
`configuration/create_log_directories.sh` creates every required job folder.
It is safe to rerun after adding or renaming launchers. Submit all documented
SBATCH commands from the repository root so those relative paths resolve
inside the checkout.

## Optional repository checks

```bash
python -m compileall -q analysis configuration curation figures processing utils
find analysis configuration curation processing figures -type f \
  \( -name '*.sh' -o -name '*.sbatch' \) -print0 \
  | xargs -0 -n1 bash -n
```
