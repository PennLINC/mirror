# Registration, warping, and quality control

Run every SBATCH command from the repository root. This makes the relative
`#SBATCH --output=logs/...` paths resolve below the checkout.

```bash
cd /cbica/projects/nibs/code_replication
micromamba activate processing
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

## T1w registration

The first replicated processing step estimates the ACPC↔T1w transforms:

```bash
registration_job=$(sbatch --parsable \
  processing/03_registration_and_warping/01_submit_t1w_registration.sbatch)
echo "T1w registration job: ${registration_job}"
```

Outputs are written to:

```text
<output_derivatives_dir>/t1w_registration/
```

## Atlas and bundle warping

Both branches depend on successful T1w registration and may then run in
parallel:

```bash
dkt_warp_job=$(sbatch --parsable --dependency="afterok:${registration_job}" \
  processing/03_registration_and_warping/02_submit_dkt_atlas_warping.sbatch)

bundle_warp_job=$(sbatch --parsable --dependency="afterok:${registration_job}" \
  processing/03_registration_and_warping/03_submit_bundle_warping.sbatch)

echo "DKT warp job: ${dkt_warp_job}"
echo "Bundle warp job: ${bundle_warp_job}"
```

These are sibling jobs with the same dependency. Slurm may run the DKT atlas
warp and bundle warp at the same time; neither waits for the other.

The DKT segmentation is stored with the registration derivatives; warped
AutoTrack bundles are stored under:

```text
<output_derivatives_dir>/warped_bundles/
```

## Quality-control reports

Coregistration and scalar reports inspect reusable inputs but write reportlets
under the replication output root:

```bash
coreg_qc_job=$(sbatch --parsable \
  processing/04_quality_control/01_coregistration_reports/submit.sbatch)

scalar_qc_job=$(sbatch --parsable \
  processing/04_quality_control/02_scalar_reports/submit.sbatch)
```

After registration, warping, and the analyses are available, generate the
consolidated spatial QC report:

```bash
sbatch processing/04_quality_control/03_analysis_qc/submit.sbatch
```

QC outputs are collected below:

```text
<output_derivatives_dir>/quality_control/
<output_derivatives_dir>/qc_report/
```

```{note}
Successful job completion is not a substitute for visual QC. Review missing
input warnings, registration overlays, scalar reportlets, and the consolidated
report before interpreting replicated statistics.
```
