# Manuscript analysis workflow

The numbered directories encode the manuscript-analysis order. Commands use
the reusable source derivatives for scalar images and the selected run output
namespace for all newly computed products.

1. `00_prepare_inputs/01_mni_ribbon_masks`: create subject-specific MNI ribbon
   masks used by GM/WM effect sizes and subject-mask voxel correlations.
2. `00_prepare_inputs/02_dkt_parcel_stats`: summarize scalar maps in DKT
   parcels after DKT warping.
3. `00_prepare_inputs/03_bundle_myelin_stats`: summarize T1w-space scalar maps
   within warped AutoTrack bundles. Its launcher uses host `tckmap` when it is
   on `PATH`; otherwise, it runs the complete participant task inside the
   configured QSIRecon container so repeated calls do not repeatedly start the
   container.
4. `01_build_missingness_list.py`: build the Figure 2
   acquisition-availability table.
5. `02_gm_wm_differentiation`: compute Figure 3 and Figure S2 effect sizes.
6. `03_correlations`: compute voxelwise and parcel/bundle correlations for
   Figure 4 and Figures S3-S6.
7. `04_icc`: compute parcel/bundle and voxelwise ICC for Figure 5 and Figures
   S7-S11.
8. `05_discriminability`: compute voxelwise and parcel/bundle discriminability
   for Table 4 and Figure S12.

The two regional-statistic preparation branches can run in parallel after the
corresponding processing products exist. ICC and discriminability branches can
also run in parallel once their shared input summaries are complete.

Every analysis launcher accepts the `ANALYSIS_SET` environment variable:

- `primary` (default) runs only the metrics used in the main analyses; and
- `full` processes the expanded metric set and writes both primary and full
  result views for a complete reproduction.

For example, `sbatch path/to/submit.sbatch` runs the primary set, while
`sbatch --export=ALL,ANALYSIS_SET=full path/to/submit.sbatch` runs the full set.
