# BIDS curation workflow

Run the numbered scripts in order. This pipeline constructs and edits the raw
BIDS dataset; it is provenance for the data release and is not part of a
manuscript analysis replication.

1. `00_download_source_data.sh`
2. `01_extract_download_archives.py`
3. `02_extract_dicom_archives.py`
4. `03_convert_dicoms_to_bids.sh` (uses `heuristic.py`)
5. `04_convert_mp2rage_phase.py`
6. `05_fix_mp2rage_phase.py`
7. `06_split_ihmt.py`
8. `07_enable_bids_writes.sh`
9. `08_anonymize_acquisition_times.py`
10. `09_clean_json_metadata.py`
11. `10_fix_bids_metadata.py`
12. `11_validate_bids.sh`
13. `12_initialize_datalad.sh`
14. `13_reface_anatomicals.sh`
15. `14_fix_mese_direction_labels.py`

The files under `status/` record completed source-download and extraction work
so interrupted curation runs can resume. They are local workflow state, not
inputs to downstream processing or analysis.

Several scripts rename or remove raw BIDS files by design. Run them only during
dataset curation and review the DataLad history after each destructive stage.
