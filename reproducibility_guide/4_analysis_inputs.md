# Prepare analysis inputs

The analysis preparation stage creates MNI ribbon masks and regional summary
tables used by multiple downstream analyses.

## MNI ribbon masks

```bash
ribbon_job=$(sbatch --parsable \
  analysis/00_prepare_inputs/01_mni_ribbon_masks/submit.sbatch)
```

These subject-specific masks are used by the GM/WM effect-size workflow and by
voxelwise correlations using subject masks. The shared mask-construction code
is in `utils/tissue_masks.py`.

## DKT parcel statistics

This step requires successful DKT atlas warping:

```bash
dkt_stats_job=$(sbatch --parsable --dependency="afterok:${dkt_warp_job}" \
  analysis/00_prepare_inputs/02_dkt_parcel_stats/submit.sbatch)
```

## AutoTrack bundle statistics

This step requires successful bundle warping:

```bash
bundle_stats_job=$(sbatch --parsable --dependency="afterok:${bundle_warp_job}" \
  analysis/00_prepare_inputs/03_bundle_myelin_stats/submit.sbatch)
```

This launcher first looks for `tckmap` in the activated environment's `PATH`.
When it is available, the task runs in that host environment. Otherwise, the
launcher uses the QSIRecon image named by `apptainer.qsirecon` in the selected
profile. It enters the container once per participant array task; the Python
program and all of its MRtrix `tckmap` calls then run within that same
container. A separate host MRtrix installation is therefore optional.

Setting `TCKMAP_BIN` explicitly takes precedence over both automatic options.

The DKT and bundle summary branches are independent and may run concurrently.

## Acquisition-availability table

Figure 2 uses a modality-availability table generated from the raw BIDS
dataset:

```bash
python analysis/01_build_missingness_list.py
```

The new table is written to:

```text
<output_derivatives_dir>/missingness/missingness_list.tsv
```

The Figure 2 script prefers this run-specific table. If it is absent, it falls
back to the checked manuscript table at `data/qc/missingness_list.tsv`.

## Expected preparation outputs

Before continuing, confirm that the selected output tree contains:

- `mni_ribbon_masks/`
- `t1w_registration/`
- `warped_bundles/`
- `DKTatlas_myelin_stats/`
- `bundle_myelin_stats/`

Exact filenames are BIDS-like and include participant, session, space, and
analysis entities where appropriate.
