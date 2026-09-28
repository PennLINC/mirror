"""Canonical repository, source-derivative, and run-output paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from configuration import load_config


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved paths for one MIRROR machine/run profile."""

    project_root: Path
    code_root: Path
    source_derivatives: Path
    output_derivatives: Path
    data: Path
    figures: Path
    run_figures: Path
    tables: Path
    run_tables: Path
    logs: Path
    work: Path
    run_name: str

    def output(self, name: str) -> Path:
        """Return a named derivative inside this run's protected output root."""

        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Output path must be relative to the run root: {name!r}')
        return self.output_derivatives / relative


def project_paths(profile: str | Path | None = None) -> ProjectPaths:
    """Load a profile as typed :class:`ProjectPaths`."""

    config = load_config(profile)
    return ProjectPaths(
        project_root=Path(config['project_root']),
        code_root=Path(config['code_dir']),
        source_derivatives=Path(config['source_derivatives_dir']),
        output_derivatives=Path(config['output_derivatives_dir']),
        data=Path(config['data_dir']),
        figures=Path(config['figures_dir']),
        run_figures=Path(config['run_figures_dir']),
        tables=Path(config['tables_dir']),
        run_tables=Path(config['run_tables_dir']),
        logs=Path(config['logs_dir']),
        work=Path(config['work_dir']),
        run_name=str(config['run_name']),
    )


# Repository assets always resolve from this checkout, even when the dataset is
# mounted elsewhere from the repository checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT

try:
    _PATHS = project_paths()
except RuntimeError as exc:
    # Keep dependency-light commands such as ``--help`` usable when PyYAML is
    # absent. Real workflows install PyYAML and therefore use the selected
    # profile; this fallback only supplies safe checkout-local defaults.
    if 'PyYAML' not in str(exc):
        raise
    _fallback_project = Path(
        os.environ.get('MIRROR_PROJECT_ROOT')
        or os.environ.get('NIBS_PROJECT_ROOT')
        or REPO_ROOT.parent
    )
    _fallback_run = (
        os.environ.get('MIRROR_RUN_NAME')
        or os.environ.get('NIBS_RUN_NAME')
        or 'mirror_data_descriptor'
    )
    _fallback_source = Path(
        os.environ.get('MIRROR_SOURCE_DERIVATIVES')
        or os.environ.get('NIBS_SOURCE_DERIVATIVES')
        or _fallback_project / 'derivatives'
    )
    _fallback_output = Path(
        os.environ.get(
            'MIRROR_OUTPUT_DERIVATIVES',
            os.environ.get('NIBS_OUTPUT_DERIVATIVES', _fallback_source / _fallback_run),
        )
    )
    _PATHS = ProjectPaths(
        project_root=_fallback_project,
        code_root=REPO_ROOT,
        source_derivatives=_fallback_source,
        output_derivatives=_fallback_output,
        data=REPO_ROOT / 'data',
        figures=REPO_ROOT / 'figures',
        run_figures=_fallback_output / 'figures',
        tables=REPO_ROOT / 'figures',
        run_tables=_fallback_output / 'figures',
        logs=_fallback_project / 'logs' / _fallback_run,
        work=_fallback_project / 'work' / _fallback_run,
        run_name=_fallback_run,
    )
PROJECT_ROOT = _PATHS.project_root
SOURCE_DERIVATIVES_ROOT = _PATHS.source_derivatives
OUTPUT_DERIVATIVES_ROOT = _PATHS.output_derivatives
FIGURES_ROOT = _PATHS.figures
RUN_FIGURES_ROOT = _PATHS.run_figures
TABLES_ROOT = _PATHS.tables
RUN_TABLES_ROOT = _PATHS.run_tables
RUN_NAME = _PATHS.run_name

# Transitional alias for scripts not yet converted. New code must explicitly
# choose SOURCE_DERIVATIVES_ROOT or OUTPUT_DERIVATIVES_ROOT.
DERIVATIVES_ROOT = SOURCE_DERIVATIVES_ROOT
