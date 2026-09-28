#!/usr/bin/env python3
"""Write an HTML table of WM bundle and DKT parcel discriminability values."""

from __future__ import annotations

import argparse
import html
import math
import sys
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover - checked after argparse handles --help
    np = None
    pd = None

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.metrics import (  # noqa: E402
    SOURCE_IMAGE_COLORS,
    source_image_display_label,
)
from utils.paths import OUTPUT_DERIVATIVES_ROOT, RUN_FIGURES_ROOT  # noqa: E402

from figures._shared.discriminability import (  # noqa: E402
    SCORE_COLUMNS,
    default_gm_input,
    default_wm_input,
    load_discriminability_table,
)


def default_icc_dir() -> Path:
    return OUTPUT_DERIVATIVES_ROOT / 'parcel_bundle_icc'


def default_wm_icc_input(icc_dir: Path, analysis_set: str, stat: str) -> Path:
    return icc_dir / f'icc_wm_bundles_{analysis_set}_{stat}.csv'


def default_gm_icc_input(icc_dir: Path, analysis_set: str, stat: str) -> Path:
    return icc_dir / f'icc_gm_parcels_{analysis_set}_{stat}.csv'


def require_dependencies() -> None:
    missing = [
        name
        for name, module in (
            ('numpy', np),
            ('pandas', pd),
        )
        if module is None
    ]
    if missing:
        raise RuntimeError(
            f'Missing required Python packages: {", ".join(missing)}. '
            'Activate the MIRROR analysis environment first.'
        )


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f'Missing input file: {path}')


def load_icc_table(path: Path, tissue: str) -> pd.DataFrame:
    require_file(path)
    table = pd.read_csv(path)
    required = {'metric_key', 'metric', 'source_image', 'ICC2_1'}
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError(f'{path} is missing required columns: {", ".join(sorted(missing))}')
    table = table.loc[table['metric_key'].astype(str) != 'ALL_METRICS'].copy()
    table['ICC2_1'] = pd.to_numeric(table['ICC2_1'], errors='coerce')
    table = table.dropna(subset=['ICC2_1'])
    table['tissue'] = tissue
    return table[['tissue', 'metric_key', 'metric', 'source_image', 'ICC2_1']]


def family_icc_order(wm_icc: Path, gm_icc: Path) -> list[str]:
    icc = pd.concat(
        [
            load_icc_table(wm_icc, 'wm'),
            load_icc_table(gm_icc, 'gm'),
        ],
        ignore_index=True,
    )
    metric_level = icc.groupby(['source_image', 'tissue', 'metric_key'], as_index=False)[
        'ICC2_1'
    ].mean()
    family_level = (
        metric_level.groupby('source_image', as_index=False)['ICC2_1']
        .mean()
        .sort_values(['ICC2_1', 'source_image'], ascending=[False, True])
    )
    return family_level['source_image'].astype(str).tolist()


def load_combined_discriminability(
    wm_input: Path, gm_input: Path, score_column: str
) -> pd.DataFrame:
    return pd.concat(
        [
            load_discriminability_table(wm_input, 'wm', score_column),
            load_discriminability_table(gm_input, 'gm', score_column),
        ],
        ignore_index=True,
    )


def table_metric_key(metric_key: object) -> str:
    key = str(metric_key)
    return 'ICVF' if key.startswith('ICVF') else key


def metric_table(discriminability: pd.DataFrame) -> pd.DataFrame:
    discriminability = discriminability.copy()
    discriminability['table_metric_key'] = discriminability['metric_key'].map(table_metric_key)
    wide = discriminability.pivot_table(
        index='table_metric_key',
        columns='tissue',
        values='score',
        aggfunc='first',
    )
    meta = (
        discriminability.sort_values(['table_metric_key', 'tissue'])
        .drop_duplicates('table_metric_key')
        .set_index('table_metric_key')
    )
    rows = wide.reset_index()
    rows = rows.rename(columns={'table_metric_key': 'metric_key'})
    rows['metric'] = rows['metric_key'].map(meta['metric']).fillna(rows['metric_key'])
    rows['source_image'] = rows['metric_key'].map(meta['source_image']).fillna('Other')
    for tissue in ('wm', 'gm'):
        if tissue not in rows:
            rows[tissue] = np.nan
    return rows[['source_image', 'metric_key', 'metric', 'wm', 'gm']]


def score_key(value: object) -> str:
    if pd.isna(value):
        return 'NA'
    return format_score(value)


def format_score(value: object) -> str:
    if pd.isna(value):
        return '&mdash;'
    numeric = float(value)
    if math.isclose(numeric, 1.0):
        return '1'
    if math.isclose(numeric, 0.0):
        return '0'
    return f'{numeric:.3f}'.rstrip('0').rstrip('.')


def display_metric_name(metric_key: object, metric: object) -> str:
    label = str(metric)
    return f'{label}*' if str(metric_key) == 'ICVF' else label


def combine_equal_score_rows(rows: pd.DataFrame) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for _, row in rows.iterrows():
        grouped[(score_key(row['wm']), score_key(row['gm']))].append(row.to_dict())

    combined = []
    for (wm_key, gm_key), group_rows in grouped.items():
        group_rows = sorted(group_rows, key=lambda item: str(item['metric']))
        combined.append(
            {
                'metric_keys': [str(item['metric_key']) for item in group_rows],
                'metrics': [
                    display_metric_name(item['metric_key'], item['metric']) for item in group_rows
                ],
                'wm': math.nan if wm_key == 'NA' else float(wm_key),
                'gm': math.nan if gm_key == 'NA' else float(gm_key),
            }
        )
    return sorted(
        combined,
        key=lambda row: (
            math.inf if pd.isna(row['wm']) else -float(row['wm']),
            math.inf if pd.isna(row['gm']) else -float(row['gm']),
            row['metrics'][0],
        ),
    )


def observed_family_order(discriminability: pd.DataFrame, ranked_families: list[str]) -> list[str]:
    observed = set(discriminability['source_image'].dropna().astype(str))
    ordered = [family for family in ranked_families if family in observed]
    ordered.extend(sorted(observed - set(ordered)))
    return ordered


def html_metric_list(metrics: list[str]) -> str:
    escaped = [html.escape(metric) for metric in metrics]
    return ',&nbsp;'.join(escaped)


def html_style(**properties: str) -> str:
    return '; '.join(f'{name.replace("_", "-")}: {value}' for name, value in properties.items())


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    token = color.strip().lstrip('#')
    if len(token) != 6:
        return (153, 153, 153)
    return tuple(int(token[index : index + 2], 16) for index in range(0, 6, 2))


def tinted_background(color: str, alpha: float = 0.12) -> str:
    red, green, blue = hex_to_rgb(color)
    tint = tuple(round(255 * (1 - alpha) + channel * alpha) for channel in (red, green, blue))
    return f'rgb({tint[0]}, {tint[1]}, {tint[2]})'


def write_html_table(
    table: pd.DataFrame,
    family_order: list[str],
    output: Path,
    title: str,
) -> None:
    rows_html: list[str] = []
    for family in family_order:
        family_rows = table.loc[table['source_image'] == family].copy()
        if family_rows.empty:
            continue
        color = SOURCE_IMAGE_COLORS.get(family, SOURCE_IMAGE_COLORS['Other'])
        background = tinted_background(color)
        header_style = html_style(
            background_color=background,
            border_top=f'2px solid {color}',
            border_bottom='1px solid #cfcfcf',
            padding='0.38rem 0.64rem',
            text_align='left',
            font_weight='bold',
        )
        rows_html.append(
            '<tr>'
            f'<th colspan="3" style="{header_style}">{html.escape(source_image_display_label(family))}</th>'
            '</tr>'
        )
        for row in combine_equal_score_rows(family_rows):
            metric_style = html_style(
                background_color=background,
                border_bottom='1px solid #dddddd',
                padding='0.42rem 0.64rem',
                vertical_align='middle',
                line_height='1.22',
            )
            score_style = html_style(
                background_color=background,
                border_bottom='1px solid #dddddd',
                padding='0.42rem 0.64rem',
                vertical_align='middle',
                text_align='right',
                width='9.8rem',
                font_variant_numeric='tabular-nums',
            )
            rows_html.append(
                f'<tr style="background:{background}">'
                f'<td class="metric-name" style="{metric_style}">{html_metric_list(row["metrics"])}</td>'
                f'<td class="score" style="{score_style}">{format_score(row["wm"])}</td>'
                f'<td class="score" style="{score_style}">{format_score(row["gm"])}</td>'
                '</tr>'
            )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  :root {{
    color-scheme: light;
    font-family: Arial, Helvetica, sans-serif;
    font-size: 16px;
  }}
  body {{
    margin: 0;
    color: #1f1f1f;
    background: #ffffff;
  }}
  table {{
    border-collapse: collapse;
    width: min(980px, 100%);
  }}
  th,
  td {{
    border-bottom: 1px solid #dddddd;
    padding: 0.42rem 0.64rem;
    vertical-align: middle;
  }}
  thead th {{
    border-bottom: 2px solid #2c2c2c;
    font-size: 0.96rem;
    text-align: left;
  }}
  .metric-name {{
    line-height: 1.22;
  }}
  .score {{
    width: 9.8rem;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}
</style>
</head>
<body>
<table>
  <thead>
    <tr>
      <th style="border-bottom:2px solid #2c2c2c; padding:0.42rem 0.64rem; text-align:left">Metric Name</th>
      <th class="score" style="border-bottom:2px solid #2c2c2c; padding:0.42rem 0.64rem; text-align:right; width:9.8rem">Discriminability (WM)</th>
      <th class="score" style="border-bottom:2px solid #2c2c2c; padding:0.42rem 0.64rem; text-align:right; width:9.8rem">Discriminability (GM)</th>
    </tr>
  </thead>
  <tbody>
    {''.join(rows_html)}
  </tbody>
</table>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    print(f'Wrote: {output}', flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-set', choices=('primary', 'full'), default='primary')
    parser.add_argument('--stat', choices=('mean', 'median'), default='median')
    parser.add_argument(
        '--distance-metric', choices=('correlation', 'euclidean'), default='correlation'
    )
    parser.add_argument('--score-column', choices=tuple(SCORE_COLUMNS), default='discriminability')
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=OUTPUT_DERIVATIVES_ROOT / 'parcel_bundle_discriminability',
    )
    parser.add_argument('--icc-dir', type=Path, default=default_icc_dir())
    parser.add_argument('--wm-input', type=Path, default=None)
    parser.add_argument('--gm-input', type=Path, default=None)
    parser.add_argument('--wm-icc-input', type=Path, default=None)
    parser.add_argument('--gm-icc-input', type=Path, default=None)
    parser.add_argument(
        '--output',
        type=Path,
        default=RUN_FIGURES_ROOT / 'table_04_discriminability' / 'Table4.html',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_dependencies()
    input_dir = args.input_dir.expanduser().resolve()
    icc_dir = args.icc_dir.expanduser().resolve()
    wm_input = (
        args.wm_input.expanduser().resolve()
        if args.wm_input
        else default_wm_input(input_dir, args.analysis_set, args.stat, args.distance_metric)
    )
    gm_input = (
        args.gm_input.expanduser().resolve()
        if args.gm_input
        else default_gm_input(input_dir, args.analysis_set, args.stat, args.distance_metric)
    )
    wm_icc = (
        args.wm_icc_input.expanduser().resolve()
        if args.wm_icc_input
        else default_wm_icc_input(icc_dir, args.analysis_set, args.stat)
    )
    gm_icc = (
        args.gm_icc_input.expanduser().resolve()
        if args.gm_icc_input
        else default_gm_icc_input(icc_dir, args.analysis_set, args.stat)
    )

    discriminability = load_combined_discriminability(wm_input, gm_input, args.score_column)
    table = metric_table(discriminability)
    family_order = observed_family_order(table, family_icc_order(wm_icc, gm_icc))
    score_label = SCORE_COLUMNS[args.score_column]
    write_html_table(
        table,
        family_order,
        args.output.expanduser().resolve(),
        title=f'WM Bundle and Cortical GM Parcel {score_label}',
    )


if __name__ == '__main__':
    main()
