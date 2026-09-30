# Registration and warping

Run every SBATCH command from the repository root. This makes the relative
`#SBATCH --output=logs/...` paths resolve below the checkout.

```bash
cd /cbica/projects/nibs/code_replication
conda activate processing
export MIRROR_CONFIG="${PWD}/configuration/profiles/replication.example.yml"
bash configuration/create_log_directories.sh
command -v python
```

The final command should print the Python executable inside the activated
`processing` environment. Submit jobs from this same shell: Slurm normally
inherits its `PATH`, allowing each launcher to resolve `python` with
`command -v`. Do not submit with `sbatch --export=NONE`. If the cluster is
configured not to export the active environment, set it explicitly before
submitting:

```bash
export PYTHON_BIN="$(command -v python)"
```

## T₁w registration

The first replicated processing step aligns the sMRIPrep native T₁w anatomy
with the QSIPrep ACPC anatomy. Its rigid transforms let the subsequent jobs
move anatomical labels into diffusion space and diffusion bundles into
native anatomical space:

```bash
sbatch processing/03_registration_and_warping/01_submit_t1w_registration.sbatch
```

Each participant receives forward and inverse HDF5 transforms
(`from-T1w_to-ACPC` and `from-ACPC_to-T1w`) under:

```text
<output_derivatives_dir>/t1w_registration/sub-<subject>/anat/
```

## Atlas and bundle warping

**Wait for every required T₁w registration task to finish successfully.**
Then submit these two independent jobs; they may run in parallel:

```bash
sbatch processing/03_registration_and_warping/02_submit_dkt_atlas_warping.sbatch

sbatch processing/03_registration_and_warping/03_submit_bundle_warping.sbatch

```

The DKT job converts the FreeSurfer DKT segmentation to NIfTI in T₁w space
and resamples its labels into ACPC space. Both `desc-DKTatlas_dseg.nii.gz`
segmentations are stored alongside the transforms in `t1w_registration/`.
They define the parcels used to summarize anatomical and diffusion metrics.

The bundle job transforms QSIRecon MSMTAutoTrack streamlines from ACPC into
T₁w space and saves `.tck` tractograms for each participant, session, and bundle.
These allow scalar maps to be summarized within the same white-matter bundles.
Outputs are stored under:

```text
<output_derivatives_dir>/warped_bundles/
```

Continue to [analysis inputs](6_analysis_inputs.md) once the relevant warp
jobs succeed. QC report instructions are in
[derivatives creation](4_derivatives_creation.md#quality-control-reports).
