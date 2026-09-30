# Data curation

The scripts in `curation/` document how the original acquisition files became
the released BIDS dataset. **This stage precedes lab replication.** The lab
replicator uses the existing dataset and derivatives, completes
[setup](2_setup.md), and begins at [registration](5_registration_and_warping.md).

## Inputs and environment

Rebuilding the dataset requires access to the original Flywheel acquisitions
and their download credentials. The released BIDS dataset does not require
repeating that download or conversion. The scripts load project paths from
the selected `MIRROR_CONFIG`; their source-download locations and acquisition
identifiers reflect the original project.

For curation, create the separate environment from the repository root:

```bash
conda env create -f environment_curation.yml
conda activate curation
```

Review the script-specific inputs before running a stage. The refacing script,
for example, takes the BIDS directory and a log directory as positional
arguments and requires AFNI and pydeface. Some stages submit Slurm jobs; wait
for those jobs to complete before continuing to a stage that uses their files.

## Ordered stages and outputs

Run the numbered scripts in order when rebuilding the source dataset. Paths
in this table are relative to `curation/`.

| Scripts | Purpose and resulting files |
| --- | --- |
| `00_download_source_data.sh` | Download the original acquisition archives from Flywheel into the project source-data tree. |
| `01_extract_download_archives.py`, `02_extract_dicom_archives.py` | Unpack downloads and nested DICOM archives for conversion. The `status/` files track completed downloads and extractions so interrupted work can resume. |
| `03_convert_dicoms_to_bids.sh`, using `heuristic.py` | Run HeuDiConv to organize the DICOM acquisitions as BIDS NIfTI images and JSON sidecars beneath the configured `bids_dir`. |
| `04_convert_mp2rage_phase.py`, `05_fix_mp2rage_phase.py` | Convert and correct MP2RAGE phase images for subsequent quantitative processing. |
| `06_split_ihmt.py` | Split ihMT acquisitions into the images used by the ihMT processing workflow. |
| `07_enable_bids_writes.sh` | Make BIDS files writable for the remaining curation edits. |
| `08_anonymize_acquisition_times.py` | Anonymize acquisition-time metadata in the BIDS dataset. |
| `09_clean_json_metadata.py`, `10_fix_bids_metadata.py` | Clean JSON sidecars and correct BIDS metadata used by downstream processing. |
| `11_validate_bids.sh` | Run BIDS validation and report dataset organization or metadata problems. |
| `12_initialize_datalad.sh` | Initialize DataLad tracking for the curated dataset. |
| `13_reface_anatomicals.sh` | Reface T₁w and deface T₂w/MP2RAGE anatomical images, replacing originals with images labeled `rec-refaced` or `rec-defaced`. |
| `14_fix_mese_direction_labels.py` | Correct MESE direction labels in the curated dataset. |

These stages edit, rename, and sometimes remove raw files as part of building
the release. Review their outputs and the DataLad history when regenerating
the dataset. The resulting BIDS tree supplies the anatomical, diffusion, and
other acquisitions used in [derivatives creation](4_derivatives_creation.md).
