"""Shared IO and QC helpers for parcel/bundle metric profile analyses."""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from utils.metric_io import add_metric_metadata, canonical_metric_from_row, canonical_metric_name
from utils.metrics import build_metric_specs, norm_token
from utils.paths import CODE_ROOT, OUTPUT_DERIVATIVES_ROOT, SOURCE_DERIVATIVES_ROOT


DEFAULT_QC_FILE = CODE_ROOT / 'data' / 'qc' / 'manual_qc_modality.tsv'
QC_MODES = ('metricqc',)
DEFAULT_WM_GLOBS = [
    str(
        SOURCE_DERIVATIVES_ROOT
        / 'qsirecon/derivatives/qsirecon-*/sub-*/ses-*/dwi/sub-*_ses-*_*_scalarstats.tsv'
    ),
    str(
        OUTPUT_DERIVATIVES_ROOT
        / 'bundle_myelin_stats/sub-*/ses-*/dwi/sub-*_ses-*_acq-HBCD75_run-01_space-T1w_model-*_scalarstats.tsv'
    ),
]
DEFAULT_DKT_GLOBS = [
    str(
        OUTPUT_DERIVATIVES_ROOT
        / 'DKTatlas_myelin_stats'
        / 'sub-*/sub-*_ses-*_run-*_desc-DKTatlas_scalarstats.csv'
    )
]
EXCLUDED_DKT_METRICS = {'G-ihMTsat', 'G-ihMTR'}
FILE_RE = re.compile(r'sub-(?P<sub>[^_]+)_(?P<ses>ses-[^_]+)_(?P<run>run-[^_]+)_')
PATH_RE = re.compile(r'sub-(?P<sub>[^_/]+).*(ses-(?P<ses>[^_/]+))')
REQUIRED_BUNDLE_COLUMNS = {'bundle', 'variable_name', 'masked_mean', 'masked_median'}
EXCLUDED_BUNDLE_PATTERNS = (
    'AnteriorCommissure',
    'DentatorubrothalamicTract-lr',
    'DentatorubrothalamicTract-rl',
    'DentatorubrothalamicTractlr',
    'DentatorubrothalamicTractrl',
)
AUTOTRACK_SPEC_FILE = CODE_ROOT / 'processing' / '01_smri_dmri' / 'qsirecon_spec.yml'
EXPECTED_WM_BUNDLE_COUNT = 91


def require_regional_analysis_inputs(
    wm_input_globs: list[str],
    dkt_input_globs: list[str],
    *,
    require_wm: bool,
    require_dkt: bool,
) -> None:
    """Fail quickly when required prepared regional summaries are absent.

    The default WM inputs combine reusable QSIRecon statistics with bundle
    summaries created by ``analysis/00_prepare_inputs``. Checking only whether
    *any* WM input exists can therefore conceal a missing bundle-summary step.
    """

    checks: list[tuple[str, list[str], str]] = []
    if require_wm:
        wm_patterns = (
            [DEFAULT_WM_GLOBS[-1]]
            if list(wm_input_globs) == DEFAULT_WM_GLOBS
            else list(wm_input_globs)
        )
        checks.append(
            (
                'prepared WM bundle statistics',
                wm_patterns,
                'analysis/00_prepare_inputs/03_bundle_myelin_stats/submit.sbatch',
            )
        )
    if require_dkt:
        checks.append(
            (
                'prepared DKT parcel statistics',
                list(dkt_input_globs),
                'analysis/00_prepare_inputs/02_dkt_parcel_stats/submit.sbatch',
            )
        )

    missing: list[str] = []
    for label, patterns, producer in checks:
        if not any(glob(pattern) for pattern in patterns):
            rendered_patterns = ', '.join(patterns)
            missing.append(
                f'{label}: no files matched [{rendered_patterns}]. Run {producer} first.'
            )
    if missing:
        raise FileNotFoundError('Missing regional analysis inputs:\n- ' + '\n- '.join(missing))


@lru_cache(maxsize=None)
def autotrack_bundle_ids(spec_file: Path = AUTOTRACK_SPEC_FILE) -> tuple[str, ...]:
    """Read the DSI Studio AutoTrack IDs directly from the QSIRecon specification."""

    if not spec_file.exists():
        raise FileNotFoundError(f'Missing QSIRecon specification: {spec_file}')
    bundle_ids: list[str] = []
    for line in spec_file.read_text().splitlines():
        match = re.match(r'^\s*track_id:\s*(.+?)\s*$', line)
        if match is None:
            continue
        value = match.group(1).strip().strip('"\'')
        bundle_ids.extend(item.strip() for item in value.split(',') if item.strip())
    bundle_ids = list(dict.fromkeys(bundle_ids))
    if not bundle_ids:
        raise RuntimeError(f'No AutoTrack track_id entries found in {spec_file}')
    return tuple(bundle_ids)


def is_excluded_wm_bundle(bundle: object) -> bool:
    text = str(bundle)
    return any(pattern in text for pattern in EXCLUDED_BUNDLE_PATTERNS)


@lru_cache(maxsize=None)
def expected_wm_bundle_ids(spec_file: Path = AUTOTRACK_SPEC_FILE) -> tuple[str, ...]:
    """Return the canonical bundle set retained by regional analyses."""

    bundles = tuple(
        bundle for bundle in autotrack_bundle_ids(spec_file) if not is_excluded_wm_bundle(bundle)
    )
    if len(bundles) != EXPECTED_WM_BUNDLE_COUNT:
        raise RuntimeError(
            f'Expected {EXPECTED_WM_BUNDLE_COUNT} retained AutoTrack bundles from {spec_file}, '
            f'but found {len(bundles)}.'
        )
    return bundles


def canonical_wm_bundle_name(
    bundle: object,
    spec_file: Path = AUTOTRACK_SPEC_FILE,
) -> str | None:
    """Map punctuation/underscore variants onto the canonical AutoTrack ID."""

    lookup = {norm_token(item): item for item in autotrack_bundle_ids(spec_file)}
    token = norm_token(bundle)
    if token in lookup:
        return lookup[token]
    if token.startswith('bundle') and token.removeprefix('bundle') in lookup:
        return lookup[token.removeprefix('bundle')]
    return None


def _normalize_subject(value: object) -> str:
    return re.sub(r'^sub-', '', str(value).strip())


def _is_pilot_subject(value: object) -> bool:
    return _normalize_subject(value).upper().startswith('PILOT')


def _session_label(value: object) -> str:
    match = re.search(r'(\d+)', str(value))
    if match is None:
        raise ValueError(f'Could not parse session number from: {value}')
    return f'Session {int(match.group(1)):02d}'


def load_qc_table(qc_file: Path) -> pd.DataFrame:
    qc_df = pd.read_csv(qc_file, sep='\t')
    if 'participant_id' not in qc_df.columns:
        raise RuntimeError(f'{qc_file} is missing participant_id')
    qc_df = qc_df.copy()
    qc_df['participant_id'] = qc_df['participant_id'].map(_normalize_subject)
    qc_df = qc_df.loc[~qc_df['participant_id'].map(_is_pilot_subject)].copy()
    return qc_df.set_index('participant_id', drop=False)


def metric_required_modalities(metric: str, patterns_file: Path) -> tuple[str, ...]:
    for spec in build_metric_specs(patterns_file):
        if metric in {spec.label, spec.primary_label}:
            return spec.qc_modalities
    raise ValueError(f'No QC modality mapping defined for metric: {metric}')


def qc_passes(
    qc_df: pd.DataFrame,
    subject: object,
    session: object,
    modalities: tuple[str, ...],
) -> bool:
    subject_id = _normalize_subject(subject)
    if subject_id not in qc_df.index:
        return False
    row = qc_df.loc[subject_id]
    session_prefix = _session_label(session)
    for modality in modalities:
        column = f'{session_prefix}--{modality}'
        if column not in qc_df.columns:
            raise RuntimeError(f'QC file is missing required column: {column}')
        value = row[column]
        if pd.isna(value) or int(value) != 1:
            return False
    return True


_qc_passes = qc_passes


def apply_metric_qc(
    df: pd.DataFrame,
    qc_df: pd.DataFrame,
    patterns_file: Path,
    subject_col: str = 'subject',
    session_col: str = 'session',
) -> pd.DataFrame:
    keep = [
        qc_passes(
            qc_df,
            row[subject_col],
            row[session_col],
            metric_required_modalities(str(row['metric']), patterns_file),
        )
        for _, row in df.iterrows()
    ]
    return df.loc[keep].copy()


def parse_dkt_filename(path: Path) -> tuple[str, str, str]:
    match = FILE_RE.search(path.name)
    if match is None:
        raise ValueError(f'Could not parse subject/session/run from: {path}')
    return match.group('sub'), match.group('ses'), match.group('run')


def collect_dkt_rows(input_glob: str) -> pd.DataFrame:
    records = []
    for file_str in sorted(glob(input_glob)):
        path = Path(file_str)
        subject, session, run = parse_dkt_filename(path)
        if _is_pilot_subject(subject):
            continue
        df = pd.read_csv(path)
        df['subject'] = subject
        df['session'] = session
        df['run'] = run
        records.append(df)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def build_dkt_value_table(
    df: pd.DataFrame,
    stat: str,
    patterns_file: Path,
) -> pd.DataFrame:
    base_cols = [
        'subject',
        'session',
        'run',
        'parcel_intensity',
        'parcel_name',
        'parcel_hemi',
        'parcel_count_t1w',
        'parcel_count_acpc',
    ]
    metric_cols = [col for col in df.columns if col.endswith(f'_{stat}')]
    if not metric_cols:
        raise RuntimeError(f'No columns found ending with _{stat}')
    value_df = df[base_cols + metric_cols].melt(
        id_vars=base_cols,
        value_vars=metric_cols,
        var_name='metric_stat',
        value_name='value',
    )
    raw_metric_names = value_df['metric_stat'].str[: -(len(stat) + 1)]
    unmapped_metrics = sorted(
        {
            metric
            for metric in raw_metric_names
            if canonical_metric_name(metric, patterns_file=patterns_file) is None
        }
    )
    value_df['metric'] = value_df['metric_stat'].str[: -(len(stat) + 1)]
    value_df = add_metric_metadata(value_df, 'metric', patterns_file)
    metric_counts = value_df['metric'].value_counts().sort_index()
    print(
        '[INFO] Loaded DKT parcel stats: '
        f'{len(df)} parcel rows, {len(metric_cols)} {stat} metric columns, '
        f'{len(value_df)} mapped long rows, {len(metric_counts)} canonical metrics.',
        flush=True,
    )
    print(
        '[INFO] DKT canonical metrics after input loading: '
        + ', '.join(f'{metric}={count}' for metric, count in metric_counts.items()),
        flush=True,
    )
    if unmapped_metrics:
        print(
            '[WARN] Dropped DKT metric columns with unmapped registry metrics: '
            + ', '.join(unmapped_metrics[:40]),
            flush=True,
        )
    value_df['parcel'] = (
        value_df['parcel_hemi'].astype(str) + '_' + value_df['parcel_name'].astype(str)
    )
    return value_df.drop(columns=['metric_stat'])


def _parse_bundle_path(path: str) -> tuple[str | None, str | None]:
    match = PATH_RE.search(path)
    if match is None:
        return None, None
    return match.group('sub'), f'ses-{match.group("ses")}'


def collect_bundle_scalarstats(input_globs: list[str], patterns_file: Path) -> pd.DataFrame:
    rows = []
    all_files = set()
    dropped_counter: Counter[tuple[str, str]] = Counter()
    for input_glob in input_globs:
        for file_path in glob(input_glob):
            all_files.add(file_path)
    for file_path in sorted(all_files):
        df = pd.read_csv(file_path, sep='\t')
        missing = REQUIRED_BUNDLE_COLUMNS.difference(df.columns)
        if missing:
            raise RuntimeError(f'Missing required columns {missing} in {file_path}')
        if 'subject_id' not in df.columns or df['subject_id'].isna().all():
            parsed_sub, _ = _parse_bundle_path(file_path)
            if parsed_sub is None:
                raise RuntimeError(f'Could not infer subject_id from {file_path}')
            df['subject_id'] = parsed_sub
        if 'session_id' not in df.columns or df['session_id'].isna().all():
            _, parsed_ses = _parse_bundle_path(file_path)
            if parsed_ses is None:
                raise RuntimeError(f'Could not infer session_id from {file_path}')
            df['session_id'] = parsed_ses
        df['source_tsv'] = file_path
        df['metric'] = df.apply(
            lambda row: canonical_metric_from_row(row, patterns_file=patterns_file),
            axis=1,
        )
        for _, drow in df[df['metric'].isna()].iterrows():
            dropped_counter[
                (str(drow.get('variable_name', '')), str(drow.get('qsirecon_suffix', '')))
            ] += 1
        df = df.dropna(subset=['metric']).copy()
        if not df.empty:
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    n_rows_before_subject_filters = len(out)
    out['subject_id'] = out['subject_id'].astype(str).str.replace('^sub-', '', regex=True)
    out = out.loc[~out['subject_id'].map(_is_pilot_subject)].copy()
    out['session_id'] = out['session_id'].astype(str)
    out['bundle'] = out['bundle'].astype(str)
    raw_bundle_names = sorted(out['bundle'].unique())
    bundle_name_map = {bundle: canonical_wm_bundle_name(bundle) for bundle in raw_bundle_names}
    unmatched = sorted(bundle for bundle, canonical in bundle_name_map.items() if canonical is None)
    if unmatched:
        print(
            '[WARN] Dropping scalarstats rows with bundle names absent from the AutoTrack '
            f'specification ({len(unmatched)} names): ' + ', '.join(unmatched[:20]),
            flush=True,
        )
    out['bundle'] = out['bundle'].map(bundle_name_map)
    out = out.dropna(subset=['bundle']).copy()
    excluded = out['bundle'].map(is_excluded_wm_bundle)
    if excluded.any():
        out = out.loc[~excluded].copy()
    observed_bundles = set(out['bundle'])
    expected_bundles = set(expected_wm_bundle_ids())
    missing_bundles = sorted(expected_bundles - observed_bundles)
    print(
        '[INFO] Canonical WM bundles after input loading: '
        f'{len(observed_bundles)}/{len(expected_bundles)} expected.',
        flush=True,
    )
    if missing_bundles:
        print(
            '[WARN] AutoTrack bundles absent from all loaded scalarstats: '
            + ', '.join(missing_bundles),
            flush=True,
        )
    metric_counts = out['metric'].value_counts().sort_index()
    print(
        '[INFO] Loaded WM scalarstats: '
        f'{len(all_files)} files, {n_rows_before_subject_filters} mapped rows before subject/bundle filters, '
        f'{len(out)} rows after filters, {len(metric_counts)} canonical metrics.',
        flush=True,
    )
    print(
        '[INFO] WM canonical metrics after input loading: '
        + ', '.join(f'{metric}={count}' for metric, count in metric_counts.items()),
        flush=True,
    )
    if dropped_counter:
        print('[WARN] Dropped rows with unmapped registry metrics (top 20):', flush=True)
        for (var_name, suffix), count in dropped_counter.most_common(20):
            print(f'  variable_name={var_name} qsirecon_suffix={suffix} n={count}', flush=True)
    return out


def apply_qc_mode(
    df: pd.DataFrame,
    qc_df: pd.DataFrame,
    qc_mode: str,
    profile_type: str,
    patterns_file: Path,
) -> pd.DataFrame:
    if profile_type in {'wm', 'dkt'}:
        if qc_mode == 'metricqc':
            return apply_metric_qc(df, qc_df, patterns_file)
    raise ValueError(f'Unsupported QC mode/profile_type: {qc_mode}/{profile_type}')


def _value_column(df: pd.DataFrame, stat: str, prefer_masked: bool) -> pd.Series:
    masked_col = f'masked_{stat}'
    if prefer_masked and masked_col in df.columns:
        masked = pd.to_numeric(df[masked_col], errors='coerce')
        raw = pd.to_numeric(df[stat], errors='coerce')
        return masked.where(np.isfinite(masked.to_numpy(dtype=float)), raw)
    return pd.to_numeric(df[stat], errors='coerce')


def load_wm_long_df(
    input_globs: list[str],
    stat: str,
    prefer_masked: bool,
    patterns_file: Path,
) -> pd.DataFrame:
    df = collect_bundle_scalarstats(input_globs, patterns_file=patterns_file)
    if df.empty:
        raise RuntimeError(f'No WM bundle scalarstats found for globs: {input_globs}')
    df = df.copy()
    df['subject'] = df['subject_id'].astype(str)
    df['session'] = df['session_id'].astype(str)
    df['feature'] = df['bundle'].astype(str)
    df['value'] = _value_column(df, stat=stat, prefer_masked=prefer_masked)
    return df[['subject', 'session', 'metric', 'feature', 'value']]


def load_dkt_long_df(
    input_globs: list[str],
    stat: str,
    patterns_file: Path,
) -> pd.DataFrame:
    row_tables = [collect_dkt_rows(input_glob) for input_glob in input_globs]
    row_tables = [table for table in row_tables if not table.empty]
    if not row_tables:
        raise RuntimeError(f'No DKT parcel stats found for glob(s): {input_globs}')
    rows = pd.concat(row_tables, ignore_index=True).drop_duplicates()
    value_df = build_dkt_value_table(rows, stat=stat, patterns_file=patterns_file)
    value_df = value_df.copy()
    value_df['feature'] = value_df['parcel'].astype(str)
    value_df = value_df.rename(columns={'subject': 'subject', 'session': 'session'})
    value_df = value_df[~value_df['metric'].isin(EXCLUDED_DKT_METRICS)].copy()
    value_df['value'] = pd.to_numeric(value_df['value'], errors='coerce')
    return value_df[['subject', 'session', 'metric', 'feature', 'value']]
