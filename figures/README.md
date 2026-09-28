# Manuscript figures and tables

Every manuscript artifact has one folder. Figure entry points follow
`plot_figure_<number>_<name>.py`; output stems are exactly `Figure1`,
`FigureS1`, and so on. Tables live here as manuscript artifacts too.

By default, generated artifacts are written below
`<output_derivatives_dir>/figures/<artifact-folder>/`. This keeps every new
artifact inside the selected run's isolated output tree and prevents a
replication from overwriting checked manuscript outputs. Set `run_figures_dir`
in a profile to choose another destination. Pass a script-level output option
only when deliberately publishing a result into the artifact's source folder.

| Artifact | Command |
| --- | --- |
| Figure 1 | `python figures/figure_01_primary_maps/plot_figure_1_primary_maps.py` |
| Figure 2 | `python figures/figure_02_missingness/plot_figure_2_missingness.py` |
| Figure 3 | `python figures/figure_03_gm_wm_effect_sizes/plot_figure_3_gm_wm_effect_sizes.py` |
| Figure 4 | `python figures/figure_04_correlations/plot_figure_4_correlations.py` |
| Figure 5 | `python figures/figure_05_icc/plot_figure_5_icc.py` |
| Figure S1 | `python figures/figure_s01_workflow/plot_figure_s1_workflow.py` |
| Figure S2 | `python figures/figure_s02_gm_wm_effect_sizes/plot_figure_s2_gm_wm_effect_sizes.py` |
| Figures S3-S6 | `python figures/figure_s03-s06_correlation_matrices/plot_figure_s3_s6_correlation_matrices.py` |
| Figure S7 | `python figures/figure_s07_variability_ratio/plot_figure_s7_variability_ratio.py` |
| Figures S8-S11 | `python figures/figure_s08-s11_icc/plot_figure_s8_s11_icc.py` |
| Figure S12 | `python figures/figure_s12_discriminability/plot_figure_s12_discriminability.py` |
| Table 4 | `python figures/table_04_discriminability/make_table_4_discriminability.py` |
| Table S3 | `python figures/table_s03_metrics/make_table_s3_metrics.py` |

Table S2 is a checked artifact derived from the QSIRecon AutoTrack bundle
specification; its former generator was already absent before this refactor.
Shared plotting implementations live in `_shared/` and are not manuscript
entry points.
