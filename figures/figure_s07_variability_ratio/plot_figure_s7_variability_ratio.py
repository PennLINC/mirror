#!/usr/bin/env python3
"""Plot regional within-/between-participant SD ratios for primary metrics."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Patch, Rectangle
except ImportError:  # pragma: no cover - checked after argparse handles --help
    mpl = None
    plt = None
    np = None
    pd = None
    Patch = None
    Rectangle = None

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.metrics import (  # noqa: E402
    METRIC_FAMILY_LEGEND_TITLE,
    SOURCE_IMAGE_COLORS,
    build_metric_specs,
    metric_display_labels,
    metric_order,
    source_image_display_label,
)
from utils.paths import CODE_ROOT, OUTPUT_DERIVATIVES_ROOT, RUN_FIGURES_ROOT  # noqa: E402


TISSUE_CONFIG = {
    'wm': ('wm_bundles', 'White Matter Bundles', 'bundle'),
    'gm': ('gm_parcels', 'Cortical Gray Matter Parcels', 'parcel'),
}


def require_dependencies() -> None:
    missing = [
        name
        for name, module in (
            ('matplotlib', mpl),
            ('numpy', np),
            ('pandas', pd),
        )
        if module is None
    ]
    if missing:
        raise RuntimeError('Missing required Python packages: ' + ', '.join(missing))


def regional_input_path(icc_dir: Path, tissue: str, stat: str) -> Path:
    profile, _, _ = TISSUE_CONFIG[tissue]
    return icc_dir / f'icc_{profile}_primary_{stat}.csv'


def load_ratio_data(
    path: Path,
    tissue: str,
    patterns_file: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Missing regional ICC input: {path}')
    table = pd.read_csv(path)
    required = {'feature', 'within_subject_sd', 'between_subject_sd'}
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError(
            f'{path} is missing {", ".join(sorted(missing))}. Rerun '
            'compute_parcel_bundle_icc.py to generate the variability diagnostics.'
        )
    if 'metric_key' not in table.columns:
        if 'metric' not in table.columns:
            raise RuntimeError(f'{path} has neither metric_key nor metric.')
        table['metric_key'] = table['metric'].astype(str)

    specs = build_metric_specs(patterns_file)
    allowed = set(metric_order(specs, 'primary', tissue=tissue))
    display = metric_display_labels(specs, 'primary', tissue=tissue)
    sources = {spec.label: spec.source_image for spec in specs}

    table = table.copy()
    table['metric_key'] = table['metric_key'].astype(str)
    table = table.loc[table['metric_key'].isin(allowed)].copy()
    within = pd.to_numeric(table['within_subject_sd'], errors='coerce')
    between = pd.to_numeric(table['between_subject_sd'], errors='coerce')
    valid = np.isfinite(within) & np.isfinite(between) & (between > 0)
    table = table.loc[valid].copy()
    table['within_between_sd_ratio'] = within.loc[valid] / between.loc[valid]
    table['tissue'] = tissue
    table['metric'] = table['metric_key'].map(display).fillna(table['metric_key'])
    table['source_image'] = table['metric_key'].map(sources).fillna('Other')
    return table[
        [
            'tissue',
            'metric_key',
            'metric',
            'source_image',
            'feature',
            'within_subject_sd',
            'between_subject_sd',
            'within_between_sd_ratio',
        ]
    ]


def summarize_ratios(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (tissue, metric_key, metric, source), group in data.groupby(
        ['tissue', 'metric_key', 'metric', 'source_image'],
        sort=False,
    ):
        values = group['within_between_sd_ratio'].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        rows.append(
            {
                'tissue': tissue,
                'metric_key': metric_key,
                'metric': metric,
                'source_image': source,
                'n_regions': int(values.size),
                'median': float(median),
                'q25': float(q25),
                'q75': float(q75),
                'mean': float(np.mean(values)),
            }
        )
    return pd.DataFrame(rows)


def ratio_axis_upper(summary: pd.DataFrame) -> float:
    finite = summary[['median', 'q75']].to_numpy(dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    maximum = max(1.0, float(np.max(finite)))
    step = 0.25 if maximum <= 2.0 else 0.5
    return float(np.ceil((maximum * 1.06) / step) * step)


def draw_ratio_panel(
    ax,
    summary: pd.DataFrame,
    tissue: str,
    upper: float,
    label_side: str,
) -> None:
    tissue_summary = summary.loc[summary['tissue'] == tissue].copy()
    tissue_summary = tissue_summary.sort_values(
        ['median', 'metric'], ascending=[True, True]
    ).reset_index(drop=True)
    positions = np.arange(len(tissue_summary))

    for position, row in tissue_summary.iterrows():
        color = SOURCE_IMAGE_COLORS.get(row['source_image'], SOURCE_IMAGE_COLORS['Other'])
        ax.add_patch(
            Rectangle(
                (row['q25'], position - 0.20),
                max(row['q75'] - row['q25'], 0.002 * upper),
                0.40,
                facecolor=color,
                edgecolor='#2b2b2b',
                linewidth=0.8,
                alpha=0.88,
                zorder=2,
            )
        )
        ax.plot(
            [row['median'], row['median']],
            [position - 0.23, position + 0.23],
            color='white',
            lw=1.8,
            zorder=3,
        )
        ax.scatter(
            [row['median']],
            [position],
            s=25,
            facecolor='white',
            edgecolor='#2b2b2b',
            linewidth=0.7,
            zorder=4,
        )
        label_x, alignment = (-0.02, 'right') if label_side == 'left' else (1.02, 'left')
        ax.text(
            label_x,
            position,
            row['metric'],
            transform=ax.get_yaxis_transform(),
            ha=alignment,
            va='center',
            fontsize=10.4,
            clip_on=False,
        )

    ax.axvline(1.0, color='#777777', lw=1.2, ls='--', zorder=1)
    ax.text(
        0.72,
        0.006,
        'Between > within',
        transform=ax.get_xaxis_transform(),
        ha='center',
        va='bottom',
        fontsize=10.2,
        color='#2F5F9E',
        fontweight='bold',
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.82, 'pad': 1.5},
        clip_on=False,
    )
    if upper > 1.0:
        ax.text(
            0.5 * (1.0 + upper),
            0.006,
            'Within > between',
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='bottom',
            fontsize=10.2,
            color='#B12A2A',
            fontweight='bold',
            bbox={
                'facecolor': 'white',
                'edgecolor': 'none',
                'alpha': 0.82,
                'pad': 1.5,
            },
            clip_on=False,
        )
    ax.set_xlim(0.0, upper)
    ax.set_ylim(-0.8, len(tissue_summary) - 0.2)
    ax.set_yticks(positions)
    ax.set_yticklabels([])
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', labelsize=10.5)
    ax.set_xlabel(
        'Within-participant test–retest SD / between-participant SD',
        fontsize=11.5,
        fontweight='bold',
        labelpad=9,
    )
    ax.set_title(
        TISSUE_CONFIG[tissue][1],
        loc='left',
        fontsize=16,
        fontweight='bold',
        pad=10,
    )
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_figure(
    data: pd.DataFrame,
    output_stem: Path,
) -> None:
    summary = summarize_ratios(data)
    if summary.empty:
        raise RuntimeError('No finite within-/between-participant SD ratios to plot.')
    upper = ratio_axis_upper(summary)

    mpl.rcParams.update(
        {
            'font.family': 'Arial',
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(18.0, 11.5), constrained_layout=False)
    draw_ratio_panel(axes[0], summary, 'wm', upper, label_side='left')
    draw_ratio_panel(axes[1], summary, 'gm', upper, label_side='right')
    fig.subplots_adjust(left=0.16, right=0.84, top=0.91, bottom=0.13, wspace=0.08)

    for label, ax in zip(('A', 'B'), axes, strict=True):
        ax.text(
            -0.18,
            1.035,
            label,
            transform=ax.transAxes,
            fontsize=20,
            fontweight='bold',
            ha='left',
            va='bottom',
            clip_on=False,
        )

    observed = set(summary['source_image'].astype(str))
    sources = [source for source in SOURCE_IMAGE_COLORS if source in observed]
    handles = [
        Patch(
            facecolor=SOURCE_IMAGE_COLORS[source],
            edgecolor='none',
            label=source_image_display_label(source),
        )
        for source in sources
    ]
    legend = fig.legend(
        handles=handles,
        title=METRIC_FAMILY_LEGEND_TITLE,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.015),
        ncol=max(1, len(handles)),
        frameon=False,
        fontsize=11.5,
        title_fontsize=12.0,
        handlelength=1.5,
        columnspacing=1.25,
    )
    legend.get_title().set_fontweight('bold')

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_stem.with_suffix('.regional_values.tsv'), sep='\t', index=False)
    summary.to_csv(output_stem.with_suffix('.summary.tsv'), sep='\t', index=False)
    for extension in ('pdf', 'png'):
        output = output_stem.with_suffix(f'.{extension}')
        fig.savefig(output, dpi=300, bbox_inches='tight', pad_inches=0.04)
        print(f'Wrote: {output}', flush=True)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stat', choices=('mean', 'median'), default='median')
    parser.add_argument(
        '--icc-dir',
        type=Path,
        default=OUTPUT_DERIVATIVES_ROOT / 'parcel_bundle_icc',
    )
    parser.add_argument(
        '--patterns-file',
        type=Path,
        default=CODE_ROOT / 'configuration' / 'metrics.yml',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=RUN_FIGURES_ROOT / 'figure_s07_variability_ratio',
    )
    parser.add_argument('--output-name', default='FigureS7')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_dependencies()
    icc_dir = args.icc_dir.expanduser().resolve()
    patterns_file = args.patterns_file.expanduser().resolve()
    tables = [
        load_ratio_data(
            regional_input_path(icc_dir, tissue, args.stat),
            tissue,
            patterns_file,
        )
        for tissue in ('wm', 'gm')
    ]
    data = pd.concat(tables, ignore_index=True)
    for tissue, tissue_data in data.groupby('tissue'):
        _, _, region_label = TISSUE_CONFIG[tissue]
        print(
            f'[INFO] {tissue.upper()}: {tissue_data["metric_key"].nunique()} primary '
            f'metrics across {tissue_data["feature"].nunique()} {region_label}s.',
            flush=True,
        )
    output_name = args.output_name
    plot_figure(data, args.output_dir.expanduser().resolve() / output_name)


if __name__ == '__main__':
    main()
