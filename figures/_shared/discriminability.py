#!/usr/bin/env python3
"""Plot parcel/bundle discriminability summaries by tissue."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Patch
except ImportError:  # pragma: no cover - checked after argparse handles --help
    mpl = None
    plt = None
    np = None
    pd = None
    Patch = None

from utils.metrics import (
    METRIC_FAMILY_LEGEND_TITLE,
    SOURCE_IMAGE_COLORS,
    source_image_display_label,
)
from utils.paths import OUTPUT_DERIVATIVES_ROOT, RUN_FIGURES_ROOT

from figures._shared.icc import color_for_source, scatter_label_layout


SCORE_COLUMNS = {
    'discriminability': 'Discriminability',
    'nearest_neighbor_accuracy': 'Nearest-neighbor accuracy',
    'mean_rank_percentile': 'Mean rank percentile',
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
        raise RuntimeError(
            'Missing required Python packages: '
            f'{", ".join(missing)}. Activate the MIRROR analysis environment first.'
        )


def source_display_label(source: str) -> str:
    return source_image_display_label(source)


def default_wm_input(outdir: Path, analysis_set: str, stat: str, distance_metric: str) -> Path:
    masked = (
        outdir
        / f'discriminability_wm_bundles_{analysis_set}_masked_preferred_{stat}_{distance_metric}.csv'
    )
    if masked.exists():
        return masked
    return outdir / f'discriminability_wm_bundles_{analysis_set}_{stat}_{distance_metric}.csv'


def default_gm_input(outdir: Path, analysis_set: str, stat: str, distance_metric: str) -> Path:
    return outdir / f'discriminability_DKTatlas_{analysis_set}_{stat}_{distance_metric}.csv'


def load_discriminability_table(path: Path, tissue: str, score_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Missing discriminability input: {path}')
    table = pd.read_csv(path)
    required = {'profile_group_key', 'profile_group', 'source_image', score_column}
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError(f'{path} is missing required columns: {", ".join(sorted(missing))}')
    table = table.loc[table['profile_group_key'].astype(str) != 'ALL_METRICS'].copy()
    table[score_column] = pd.to_numeric(table[score_column], errors='coerce')
    table = table.dropna(subset=[score_column])
    table['tissue'] = tissue
    return table.rename(
        columns={
            'profile_group_key': 'metric_key',
            'profile_group': 'metric',
            score_column: 'score',
        }
    )[
        [
            'tissue',
            'metric_key',
            'metric',
            'source_image',
            'score',
            'n_subjects',
            'n_sessions',
            'n_profiles',
            'n_features',
        ]
    ]


def metric_order(data: pd.DataFrame, tissue: str) -> list[str]:
    tissue_data = data.loc[data['tissue'] == tissue].copy()
    tissue_data = tissue_data.sort_values(['score', 'metric'], ascending=[True, True])
    return tissue_data['metric_key'].tolist()


def draw_bar_panel(ax, data: pd.DataFrame, tissue: str, title: str, xlabel: str) -> None:
    order = metric_order(data, tissue)
    tissue_data = data.loc[data['tissue'] == tissue].set_index('metric_key')
    if not order:
        ax.text(0.5, 0.5, f'No {title} data', ha='center', va='center')
        ax.set_axis_off()
        return
    positions = np.arange(len(order))
    for position, metric_key in zip(positions, order):
        row = tissue_data.loc[metric_key]
        color = color_for_source(row['source_image'])
        ax.barh(
            position,
            row['score'],
            height=0.52,
            color=color,
            edgecolor='none',
            alpha=0.88,
            zorder=2,
        )
        ax.text(
            -0.02,
            position,
            row['metric'],
            transform=ax.get_yaxis_transform(),
            ha='right',
            va='center',
            fontsize=8.5,
            clip_on=False,
        )
        ax.text(
            row['score'] + 0.012,
            position,
            f'{row["score"]:.2f}',
            ha='left',
            va='center',
            fontsize=8.2,
            color='black',
        )
    for value in (0.5, 0.75, 0.9):
        ax.axvline(value, color='#d7d7d7', lw=0.9, zorder=0)
    ax.set_yticks(positions)
    ax.set_yticklabels([])
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlim(0.0, 1.03)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc='left', fontweight='bold')
    ax.grid(False)
    ax.set_box_aspect(1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def matched_scores(data: pd.DataFrame) -> pd.DataFrame:
    wide = data.pivot_table(index='metric_key', columns='tissue', values='score', aggfunc='first')
    return wide.dropna(subset=['wm', 'gm'])


def draw_scatter_panel(ax, data: pd.DataFrame, xlabel: str, ylabel: str) -> None:
    wide = matched_scores(data)
    meta = data.drop_duplicates('metric_key').set_index('metric_key')
    if wide.empty:
        ax.text(0.5, 0.5, 'No matched WM/GM metrics', ha='center', va='center')
        ax.set_axis_off()
        return
    lower, upper = 0.0, 1.0
    identity = np.linspace(lower, upper, 200)
    ax.fill_between(identity, lower, identity, color='#eeeeee', zorder=0)
    ax.plot(identity, identity, color='#8c8c8c', lw=1.0, ls='--', zorder=1)
    for value in (0.5, 0.75, 0.9):
        ax.axvline(value, color='#e0e0e0', lw=0.8, zorder=0)
        ax.axhline(value, color='#e0e0e0', lw=0.8, zorder=0)

    label_positions = scatter_label_layout(wide, meta, lower, upper)
    for metric_key, row in wide.iterrows():
        source = meta.loc[metric_key, 'source_image']
        color = color_for_source(source)
        label = meta.loc[metric_key, 'metric']
        label_x, label_y, label_ha, use_leader = label_positions[metric_key]
        ax.scatter(
            row['gm'],
            row['wm'],
            s=44,
            facecolor=color,
            edgecolor='#2b2b2b',
            linewidth=0.7,
            alpha=0.95,
            zorder=3,
        )
        if use_leader:
            ax.annotate(
                label,
                xy=(row['gm'], row['wm']),
                xytext=(label_x, label_y),
                textcoords='data',
                arrowprops={
                    'arrowstyle': '-',
                    'color': color,
                    'alpha': 0.55,
                    'lw': 0.6,
                    'shrinkA': 1,
                    'shrinkB': 4,
                },
                fontsize=7.2,
                color=color,
                ha=label_ha,
                va='center',
                zorder=4,
            )
        else:
            ax.text(
                label_x,
                label_y,
                label,
                fontsize=7.2,
                color=color,
                ha=label_ha,
                va='center',
                zorder=4,
            )
    ax.text(
        0.10,
        0.82,
        'WM > GM',
        transform=ax.transAxes,
        color='#6a6a6a',
        fontsize=10,
        fontstyle='italic',
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.80, 'pad': 1.0},
    )
    ax.text(
        0.70,
        0.13,
        'GM > WM',
        transform=ax.transAxes,
        color='#6a6a6a',
        fontsize=10,
        fontstyle='italic',
        bbox={'facecolor': '#eeeeee', 'edgecolor': 'none', 'alpha': 0.80, 'pad': 1.0},
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect('equal', adjustable='box')
    ax.set_box_aspect(1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def add_source_legend(fig, data: pd.DataFrame) -> None:
    sources = [source for source in SOURCE_IMAGE_COLORS if source in set(data['source_image'])]
    handles = [
        Patch(
            facecolor=color_for_source(source), edgecolor='none', label=source_display_label(source)
        )
        for source in sources
    ]
    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=len(handles),
        frameon=False,
        title=METRIC_FAMILY_LEGEND_TITLE,
        bbox_to_anchor=(0.5, 0.014),
        fontsize=10.5,
        title_fontsize=11,
    )


def plot_discriminability(data: pd.DataFrame, output_prefix: Path, score_label: str) -> None:
    if data.empty:
        raise RuntimeError('No discriminability data available.')
    mpl.rcParams.update(
        {
            'font.family': 'Arial',
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'axes.titlesize': 14.5,
            'axes.labelsize': 11.5,
            'xtick.labelsize': 9.6,
            'ytick.labelsize': 9.2,
        }
    )
    fig = plt.figure(figsize=(8.8, 18.2), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.0, 1.0, 1.05],
        left=0.245,
        right=0.86,
        bottom=0.085,
        top=0.985,
        hspace=0.18,
    )
    axes = [fig.add_subplot(grid[index, 0]) for index in range(3)]
    draw_bar_panel(
        axes[0], data, 'wm', 'White Matter (Bundles)', f'{score_label} across WM bundles'
    )
    draw_bar_panel(axes[1], data, 'gm', 'Gray Matter (Parcels)', f'{score_label} across GM parcels')
    draw_scatter_panel(
        axes[2],
        data,
        xlabel=f'{score_label} across GM parcels',
        ylabel=f'{score_label} across WM bundles',
    )
    for label, ax in zip(('A', 'B', 'C'), axes):
        ax.text(
            -0.18,
            1.03,
            label,
            transform=ax.transAxes,
            fontsize=18,
            fontweight='bold',
            ha='left',
            va='bottom',
            clip_on=False,
        )
    add_source_legend(fig, data)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'png'):
        out_file = output_prefix.with_suffix(f'.{extension}')
        fig.savefig(out_file, dpi=300)
        print(f'Wrote: {out_file}', flush=True)
    data.to_csv(output_prefix.with_suffix('.summary.tsv'), sep='\t', index=False)
    plt.close(fig)


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
    parser.add_argument('--wm-input', type=Path, default=None)
    parser.add_argument('--gm-input', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=RUN_FIGURES_ROOT / 'discriminability')
    parser.add_argument('--output-name', default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_dependencies()
    input_dir = args.input_dir.expanduser().resolve()
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
    data = pd.concat(
        [
            load_discriminability_table(wm_input, 'wm', args.score_column),
            load_discriminability_table(gm_input, 'gm', args.score_column),
        ],
        ignore_index=True,
    )
    output_name = (
        args.output_name
        or f'discriminability_parcel_bundle_{args.analysis_set}_{args.stat}_{args.distance_metric}_{args.score_column}'
    )
    plot_discriminability(
        data,
        args.output_dir.expanduser().resolve() / output_name,
        SCORE_COLUMNS[args.score_column],
    )


if __name__ == '__main__':
    main()
