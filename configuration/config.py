"""Load and validate MIRROR machine and run configuration profiles.

The configuration deliberately separates reusable source derivatives from
outputs produced by this repository. Source-scalar datasets (for example,
``pymp2rage`` and ``qsm``) are resolved below ``source_derivatives_dir``;
registration, regional summaries, analyses, and manuscript artifacts are
written below ``output_derivatives_dir`` for the selected run.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_PROFILE = 'hpc'
CONFIG_ENVIRONMENT_VARIABLE = 'MIRROR_CONFIG'
LEGACY_CONFIG_ENVIRONMENT_VARIABLE = 'NIBS_CONFIG'
RUN_ENVIRONMENT_VARIABLE = 'MIRROR_RUN_NAME'
LEGACY_RUN_ENVIRONMENT_VARIABLE = 'NIBS_RUN_NAME'


def _resolve_path(base: Path, value: str | os.PathLike[str]) -> Path:
    """Resolve *value* relative to *base* while preserving absolute paths."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _profile_path(config: str | os.PathLike[str] | None) -> Path:
    """Resolve a profile name or YAML path."""

    requested = (
        config
        or os.environ.get(CONFIG_ENVIRONMENT_VARIABLE)
        or os.environ.get(LEGACY_CONFIG_ENVIRONMENT_VARIABLE)
        or DEFAULT_PROFILE
    )
    requested_path = Path(requested).expanduser()
    if requested_path.suffix.lower() in {'.yml', '.yaml'} or requested_path.is_absolute():
        if requested_path.is_file():
            return requested_path.resolve()
        raise FileNotFoundError(f'Configuration profile not found: {requested_path}')

    config_dir = Path(__file__).resolve().parent
    candidates = (
        config_dir / 'profiles' / f'{requested}.yml',
        # Transitional fallback used by existing downstream profiles and tests.
        config_dir / f'paths_{requested}.yml',
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ', '.join(str(path) for path in candidates)
    raise FileNotFoundError(f'Configuration profile {requested!r} not found; searched: {searched}')


def _resolve_mapping(base: Path, values: Mapping[str, Any]) -> dict[str, str]:
    """Resolve every path-valued item in a mapping."""

    return {key: str(_resolve_path(base, value)) for key, value in values.items()}


def load_config(config: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load a machine/run profile and return resolved absolute paths.

    Parameters
    ----------
    config
        Profile name (for example ``"hpc"``), YAML path, or ``None``.
        When omitted, ``MIRROR_CONFIG`` is used, followed by the generic
        ``hpc`` profile. ``NIBS_CONFIG`` remains a compatibility alias.

    Returns
    -------
    dict
        Backwards-compatible dictionary of resolved paths. In addition to the
        historical keys, the mapping exposes ``source_derivatives_dir``,
        ``output_derivatives_dir``, ``run_name``, ``figures_dir``,
        ``run_figures_dir``, ``tables_dir``, and ``logs_dir``.
    """

    profile_path = _profile_path(config)
    profile_text = profile_path.read_text(encoding='utf-8')
    try:
        raw = json.loads(profile_text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                f'{profile_path} uses YAML syntax; install PyYAML or use JSON-formatted YAML.'
            ) from exc
        raw = yaml.safe_load(profile_text) or {}
    if not isinstance(raw, dict):
        raise TypeError(f'Configuration must be a YAML mapping: {profile_path}')
    if 'project_root' not in raw:
        raise KeyError(f'Configuration is missing project_root: {profile_path}')

    project_root = Path(raw['project_root']).expanduser()
    if not project_root.is_absolute():
        project_root = (profile_path.parent / project_root).resolve()

    code_value = raw.get('code_dir', 'auto')
    code_dir = (
        Path(__file__).resolve().parents[1]
        if code_value in {None, 'auto'}
        else _resolve_path(project_root, code_value)
    )
    source_derivatives_dir = _resolve_path(
        project_root,
        raw.get('source_derivatives_dir', 'derivatives'),
    )
    run_name = str(
        os.environ.get(RUN_ENVIRONMENT_VARIABLE)
        or os.environ.get(LEGACY_RUN_ENVIRONMENT_VARIABLE)
        or raw.get('run_name', 'mirror_data_descriptor')
    ).strip()
    if not run_name or run_name in {'.', '..'} or '/' in run_name or '\\' in run_name:
        raise ValueError(f'Invalid run_name: {run_name!r}')

    output_derivatives_dir = _resolve_path(
        project_root,
        raw.get('output_derivatives_dir', source_derivatives_dir / run_name),
    )
    if output_derivatives_dir == source_derivatives_dir:
        raise ValueError(
            'output_derivatives_dir must differ from source_derivatives_dir to protect '
            'reusable preprocessed data.'
        )

    config_dict: dict[str, Any] = {
        'profile_file': str(profile_path),
        'project_root': str(project_root),
        'bids_dir': str(_resolve_path(project_root, raw.get('bids_dir', 'dset'))),
        'code_dir': str(code_dir),
        'data_dir': str(_resolve_path(code_dir, raw.get('data_dir', 'data'))),
        'work_dir': str(_resolve_path(project_root, raw.get('work_dir', 'work'))),
        'source_derivatives_dir': str(source_derivatives_dir),
        'output_derivatives_dir': str(output_derivatives_dir),
        'run_name': run_name,
        'figures_dir': str(_resolve_path(code_dir, raw.get('figures_dir', 'figures'))),
        'logs_dir': str(
            code_dir / 'logs'
            if raw.get('logs_dir', 'auto') in {None, 'auto'}
            else _resolve_path(project_root, raw['logs_dir'])
        ),
    }
    run_figures_value = raw.get('run_figures_dir')
    config_dict['run_figures_dir'] = str(
        _resolve_path(project_root, run_figures_value)
        if run_figures_value is not None
        else output_derivatives_dir / 'figures'
    )
    # Table generators live beside figure generators. Keep these aliases so
    # older call sites resolve to the same manuscript-artifact root.
    config_dict['tables_dir'] = config_dict['figures_dir']
    config_dict['run_tables_dir'] = str(
        _resolve_path(project_root, raw['run_tables_dir'])
        if 'run_tables_dir' in raw
        else config_dict['run_figures_dir']
    )

    # New profiles define derivative paths relative to source_derivatives_dir.
    # Older profiles defined them relative to project_root.
    derivative_base = source_derivatives_dir if 'source_derivatives_dir' in raw else project_root
    config_dict['derivatives'] = _resolve_mapping(
        derivative_base,
        raw.get('derivatives', {}),
    )

    for section in ('sourcedata', 'apptainer', 'freesurfer', 'software'):
        if section in raw:
            config_dict[section] = _resolve_mapping(project_root, raw[section])
    if 'docker' in raw:
        config_dict['docker'] = dict(raw['docker'])
    config_dict['synthstrip_runtime'] = raw.get('synthstrip_runtime', 'apptainer')

    return config_dict


def output_derivative(config: Mapping[str, Any], name: str) -> Path:
    """Return a named output directory inside the configured run namespace."""

    path = Path(name)
    if not name or name in {'.', '..'} or path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Invalid output derivative name: {name!r}')
    return Path(config['output_derivatives_dir']) / path
