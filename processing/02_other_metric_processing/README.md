# Source-scalar generation

These workflows create reusable derivatives. They are retained to document the
full data-generation pipeline, but a manuscript replication should consume
their existing outputs rather than rerun them.

## Dependency order

1. Run `process_mp2rage.py` and `process_mese.py` after sMRIPrep. They are
   independent and may run in parallel.
2. After MP2RAGE, run `process_ihmt.py` and `process_t1wt2w_ratio.py` in
   parallel.
3. After MESE, run `process_megre.py`.
4. After MEGRE, run `process_qsm.py` and `process_q_ratio.py`; Q-Ratio also
   requires MP2RAGE. Then run `process_qsm_post.py` after QSM finishes.
5. After QSIRecon and ihMT, run `process_g_ratio_scaling_factors.py` for every
   participant. Run `aggregate_g_ratio_scaling_factors.py` after all participant
   jobs finish, then run `process_g_ratio.py` with the finalized factors.

Each modality directory contains its own `submit.sbatch`, so independent
branches can run concurrently. Submit downstream branches only after the
dependencies listed above complete. The g-ratio directory instead has three
numbered launchers because its aggregation step must run between its two
subject-array stages.

All entry points can be called from the repository root, for example:

```bash
python processing/02_other_metric_processing/01_mp2rage/process_mp2rage.py \
  --subject-id 01
```

The Python workflows load paths from the shared configuration and use common
helpers from `utils.processing`.
