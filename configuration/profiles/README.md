# Configuration profiles

Profiles separate reusable preprocessing/source-scalar derivatives from the
outputs of a particular manuscript or replication run.

The default is the generic cluster profile, `hpc.yml`. Set `MIRROR_CONFIG` to
either a profile name (`hpc`, `pc`) or an absolute YAML path. Set
`MIRROR_RUN_NAME` only when you intentionally want to override the
profile's run name; normally the profile should define both `run_name` and
`output_derivatives_dir` explicitly.

Legacy `NIBS_CONFIG` and `NIBS_RUN_NAME` variables remain supported while
existing launch commands migrate to the MIRROR name.

Use `code_dir: auto` to resolve repository assets from the active checkout.
This is the default and remains valid if the clone is renamed from `nibs` to
`MIRROR` or placed anywhere else beneath the project root.

Use `logs_dir: auto` to place Slurm output and workflow job records in the
checkout's ignored `logs/` directory. Run
`configuration/create_log_directories.sh` once after cloning, then submit
SBATCH launchers from the repository root; their relative output paths require
that working directory.

For the documented CUBIC reproduction, edit `replication.example.yml` in
place. It is preconfigured for a checkout at
`/cbica/projects/nibs/code_replication`, reusable derivatives at
`/cbica/projects/nibs/derivatives`, and new outputs at
`/cbica/projects/nibs/derivatives/replication`. Because the profile is tracked,
local edits will appear in `git status`; do not commit cluster- or user-specific
changes unintentionally. Never point `output_derivatives_dir` at
`source_derivatives_dir`.

Generated figures and tables default to `<output_derivatives_dir>/figures`.
Set `run_figures_dir` (and, if desired, `run_tables_dir`) only when those
artifacts should live elsewhere.
