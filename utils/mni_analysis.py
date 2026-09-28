#!/usr/bin/env python3
"""Shared subject discovery, QC, metric-path, and robust-stat helpers."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover - calling scripts report dependencies
    np = None
    pd = None

from utils.metrics import MetricSpec, flatten_metric_patterns


def normalize_subject(value: str) -> str:
    token = str(value).strip()
    return token if token.startswith('sub-') else f'sub-{token}'


def normalize_session(value: str) -> str:
    token = str(value).strip()
    return token if token.startswith('ses-') else f'ses-{token}'


def subject_for_qc(subject: object) -> str:
    return re.sub(r'^sub-', '', str(subject).strip())


def is_pilot_subject(subject: object) -> bool:
    return subject_for_qc(subject).upper().startswith('PILOT')


def session_label(session: object) -> str:
    match = re.search(r'(\d+)', str(session))
    if match is None:
        raise ValueError(f'Could not parse session number from {session}')
    return f'Session {int(match.group(1)):02d}'


def load_patterns(path: Path) -> dict[str, str]:
    """Load flattened patterns from the authoritative metric registry."""

    return flatten_metric_patterns(path)


def first_glob(patterns: Iterable[Path]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(pattern.parent.glob(pattern.name)))
    return sorted(set(matches))[0] if matches else None


def pattern_path(
    derivatives: Path,
    rel_pattern: str,
    subject: str,
    session: str,
    space: str,
) -> Path:
    rel_pattern = rel_pattern.replace('_space-MNI152NLin2009cAsym_', '_space-{space}_')
    return derivatives / rel_pattern.format(subject=subject, session=session, space=space)


def discover_subjects(derivatives: Path) -> list[str]:
    roots = (
        derivatives / 'smriprep',
        derivatives / 'qsirecon' / 'derivatives' / 'qsirecon-DIPYDKI',
        derivatives / 'pymp2rage',
        derivatives / 'ihmt',
    )
    return sorted(
        {
            path.name
            for root in roots
            if root.is_dir()
            for path in root.glob('sub-*')
            if path.is_dir() and not is_pilot_subject(path.name)
        }
    )


def discover_sessions(derivatives: Path, subject: str) -> list[str]:
    roots = (
        derivatives / 'qsirecon' / 'derivatives' / 'qsirecon-DIPYDKI' / subject,
        derivatives / 'pymp2rage' / subject,
        derivatives / 'ihmt' / subject,
    )
    return sorted(
        {
            path.name
            for root in roots
            if root.is_dir()
            for path in root.glob('ses-*')
            if path.is_dir()
        }
    )


def load_qc_table(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    qc = pd.read_csv(path, sep='\t')
    qc['participant_id'] = qc['participant_id'].map(subject_for_qc)
    qc = qc.loc[~qc['participant_id'].map(is_pilot_subject)].copy()
    return qc.set_index('participant_id', drop=False)


def qc_passes(
    qc: pd.DataFrame | None,
    subject: str,
    session: str,
    spec: MetricSpec,
) -> bool:
    if qc is None:
        return True
    if not spec.qc_modalities:
        warnings.warn(f'No QC modality mapping for {spec.label}; applying no modality QC')
        return True
    subject_id = subject_for_qc(subject)
    if subject_id not in qc.index:
        return False
    row = qc.loc[subject_id]
    prefix = session_label(session)
    for modality in spec.qc_modalities:
        column = f'{prefix}--{modality}'
        if column not in qc.columns:
            raise RuntimeError(f'QC file is missing required column: {column}')
        value = row[column]
        if pd.isna(value) or int(value) != 1:
            return False
    return True


def robust_outlier_mask(values: np.ndarray, z_threshold: float) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return finite
    clean = values[finite]
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    if np.isfinite(mad) and mad > 0:
        robust_z = 0.67448975 * (values - median) / mad
        return finite & (np.abs(robust_z) <= z_threshold)
    q_low, q_high = np.percentile(clean, [0.1, 99.9])
    return finite & (values >= q_low) & (values <= q_high)


def metric_paths_for_session(
    derivatives: Path,
    patterns: dict[str, str],
    specs: list[MetricSpec],
    qc: pd.DataFrame | None,
    subject: str,
    session: str,
    space: str,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for spec in specs:
        if not qc_passes(qc, subject, session, spec):
            continue
        rel_pattern = patterns.get(spec.pattern_key)
        if rel_pattern is None:
            warnings.warn(f'No {space} pattern found for {spec.label}: {spec.pattern_key}')
            continue
        path = first_glob((pattern_path(derivatives, rel_pattern, subject, session, space),))
        if path is not None:
            paths[spec.label] = path
    return paths
