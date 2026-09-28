#!/usr/bin/env python3
"""Plot primary voxelwise cortical-GM/WM and parcel/bundle ICC panels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

from utils.metrics import METRIC_FAMILY_LEGEND_TITLE, SOURCE_IMAGE_COLORS  # noqa: E402
from utils.paths import CODE_ROOT, RUN_FIGURES_ROOT  # noqa: E402

from figures._shared.icc import (  # noqa: E402
    BENCHMARKS,
    color_for_source,
    default_mni_icc_dir,
    default_parcel_bundle_icc_dir,
    load_parcel_bundle_icc,
    load_voxelwise_icc,
    require_dependencies,
    scatter_label_layout,
    source_display_label,
    summarize_metric_values,
)


TISSUE_DOMAIN_TITLES = {
    ('wm', 'voxels'): 'White Matter Voxels',
    ('wm', 'bundles'): 'White Matter Bundles',
    ('gm', 'voxels'): 'Cortical Gray Matter Voxels',
    ('gm', 'parcels'): 'Cortical Gray Matter Parcels',
}


def metric_order_for_tissue(summary: pd.DataFrame, tissue: str) -> list[str]:
    tissue_summary = summary.loc[summary['tissue'] == tissue].copy()
    tissue_summary = tissue_summary.sort_values(['median', 'metric'], ascending=[True, True])
    return tissue_summary['metric_key'].tolist()


def draw_interval_panel_no_numbers(
    ax,
    summary: pd.DataFrame,
    tissue: str,
    domain: str,
    xlabel: str,
    label_side: str = 'left',
) -> None:
    order = metric_order_for_tissue(summary, tissue)
    tissue_summary = summary.loc[summary['tissue'] == tissue].set_index('metric_key')
    if not order:
        ax.text(0.5, 0.5, f'No {tissue.upper()} ICC data', ha='center', va='center')
        ax.set_axis_off()
        return

    positions = np.arange(len(order))
    for position, metric_key in zip(positions, order):
        row = tissue_summary.loc[metric_key]
        color = color_for_source(row['source_image'])
        median = row['median']
        q25 = row['q25']
        q75 = row['q75']
        ax.add_patch(
            Rectangle(
                (q25, position - 0.20),
                max(q75 - q25, 0.001),
                0.40,
                facecolor=color,
                edgecolor='#2b2b2b',
                linewidth=0.8,
                alpha=0.86,
                zorder=2,
            )
        )
        ax.plot(
            [median, median], [position - 0.23, position + 0.23], color='white', lw=1.8, zorder=3
        )
        ax.scatter([median], [position], s=24, facecolor='white', edgecolor='#2b2b2b', zorder=4)
        if label_side == 'right':
            label_x = 1.02
            label_ha = 'left'
        else:
            label_x = -0.02
            label_ha = 'right'
        ax.text(
            label_x,
            position,
            row['metric'],
            transform=ax.get_yaxis_transform(),
            ha=label_ha,
            va='center',
            fontsize=10.6,
            clip_on=False,
        )

    for benchmark in BENCHMARKS:
        ax.axvline(
            benchmark,
            color='#c7c7c7',
            lw=0.9 if benchmark else 1.1,
            ls='-' if benchmark else ':',
            zorder=0,
        )
    ax.set_yticks(positions)
    ax.set_yticklabels([])
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_title(TISSUE_DOMAIN_TITLES[(tissue, domain)], loc='left', fontweight='bold')
    ax.grid(False)
    ax.set_box_aspect(1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def matched_scatter_values(summary: pd.DataFrame) -> pd.DataFrame:
    wide = summary.pivot_table(
        index='metric_key',
        columns='tissue',
        values='median',
        aggfunc='first',
    )
    return wide.dropna(subset=['wm', 'gm'])


def common_scatter_limits(*summaries: pd.DataFrame) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for summary in summaries:
        wide = matched_scatter_values(summary)
        if not wide.empty:
            values.extend([wide['gm'].to_numpy(float), wide['wm'].to_numpy(float)])
    if not values:
        return 0.2, 1.0
    finite = np.concatenate(values)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.2, 1.0
    lower = max(0.0, float(np.nanmin(finite)) - 0.04)
    lower = min(0.2, lower)
    return lower, 1.0


def draw_scatter_panel_fixed(
    ax,
    summary: pd.DataFrame,
    gm_domain: str,
    wm_domain: str,
    lower: float,
    upper: float,
) -> None:
    wide = matched_scatter_values(summary)
    meta = summary.drop_duplicates('metric_key').set_index('metric_key')
    gm_label = 'GM'
    if wide.empty:
        ax.text(0.5, 0.5, 'No matched WM/GM metrics', ha='center', va='center')
        ax.set_axis_off()
        return

    identity = np.linspace(lower, upper, 200)
    ax.fill_between(identity, lower, identity, color='#eeeeee', zorder=0)
    ax.plot(identity, identity, color='#8c8c8c', lw=1.0, ls='--', zorder=1)
    for benchmark in BENCHMARKS:
        if lower <= benchmark <= upper:
            ax.axvline(benchmark, color='#e0e0e0', lw=0.8, zorder=0)
            ax.axhline(benchmark, color='#e0e0e0', lw=0.8, zorder=0)

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
                fontsize=9.0,
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
                fontsize=9.0,
                color=color,
                ha=label_ha,
                va='center',
                zorder=4,
            )

    ax.text(
        0.12,
        0.81,
        f'WM ICC > {gm_label} ICC',
        transform=ax.transAxes,
        color='#6a6a6a',
        fontsize=12.5,
        fontstyle='italic',
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.80, 'pad': 1.0},
    )
    ax.text(
        0.66,
        0.13,
        f'{gm_label} ICC > WM ICC',
        transform=ax.transAxes,
        color='#6a6a6a',
        fontsize=12.5,
        fontstyle='italic',
        bbox={'facecolor': '#eeeeee', 'edgecolor': 'none', 'alpha': 0.80, 'pad': 1.0},
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect('equal', adjustable='box')
    ax.set_box_aspect(1)
    ax.set_xlabel(f'Median ICC across {gm_domain}')
    ax.set_ylabel(f'Median ICC across {wm_domain}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def add_single_source_legend(fig, *dataframes: pd.DataFrame) -> None:
    observed = set()
    for data in dataframes:
        if not data.empty and 'source_image' in data.columns:
            observed.update(data['source_image'].dropna().astype(str))
    sources = [source for source in SOURCE_IMAGE_COLORS if source in observed]
    if not sources:
        return
    handles = [
        Patch(
            facecolor=color_for_source(source),
            edgecolor='none',
            label=source_display_label(source),
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
        fontsize=13.0,
        title_fontsize=13.5,
    )


def plot_combined_icc(
    voxel_data: pd.DataFrame,
    parcel_data: pd.DataFrame,
    output_prefix: Path,
) -> None:
    if voxel_data.empty:
        raise RuntimeError('No voxelwise ICC data available.')
    if parcel_data.empty:
        raise RuntimeError('No parcel/bundle ICC data available.')

    voxel_data = voxel_data.copy()
    parcel_data = parcel_data.copy()
    voxel_data['icc'] = pd.to_numeric(voxel_data['icc'], errors='coerce')
    parcel_data['icc'] = pd.to_numeric(parcel_data['icc'], errors='coerce')
    voxel_data = voxel_data.dropna(subset=['icc'])
    parcel_data = parcel_data.dropna(subset=['icc'])
    voxel_summary = summarize_metric_values(voxel_data)
    parcel_summary = summarize_metric_values(parcel_data)
    if voxel_summary.empty or parcel_summary.empty:
        raise RuntimeError('No finite ICC values available for the combined figure.')

    mpl.rcParams.update(
        {
            'font.family': 'Arial',
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'axes.titlesize': 17.5,
            'axes.labelsize': 14.0,
            'xtick.labelsize': 12.0,
            'ytick.labelsize': 11.8,
        }
    )
    fig = plt.figure(figsize=(18.2, 18.8), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.0, 1.05],
        left=0.078,
        right=0.965,
        bottom=0.085,
        top=0.955,
        hspace=0.20,
        wspace=0.0,
    )
    axes = {
        'A': fig.add_subplot(grid[0, 0]),
        'B': fig.add_subplot(grid[0, 1]),
        'C': fig.add_subplot(grid[1, 0]),
        'D': fig.add_subplot(grid[1, 1]),
        'E': fig.add_subplot(grid[2, 0]),
        'F': fig.add_subplot(grid[2, 1]),
    }

    draw_interval_panel_no_numbers(
        axes['A'], voxel_summary, 'wm', 'voxels', 'ICC(2,1) across WM voxels'
    )
    draw_interval_panel_no_numbers(
        axes['B'],
        parcel_summary,
        'wm',
        'bundles',
        'ICC(2,1) across WM bundles',
    )
    draw_interval_panel_no_numbers(
        axes['C'],
        voxel_summary,
        'gm',
        'voxels',
        'ICC(2,1) across cortical GM voxels',
    )
    draw_interval_panel_no_numbers(
        axes['D'],
        parcel_summary,
        'gm',
        'parcels',
        'ICC(2,1) across cortical GM parcels',
    )

    lower, upper = common_scatter_limits(voxel_summary, parcel_summary)
    draw_scatter_panel_fixed(
        axes['E'],
        voxel_summary,
        'cortical GM voxels',
        'WM voxels',
        lower,
        upper,
    )
    draw_scatter_panel_fixed(
        axes['F'], parcel_summary, 'cortical GM parcels', 'WM bundles', lower, upper
    )

    for label, ax in axes.items():
        ax.text(
            -0.20,
            1.015,
            label,
            transform=ax.transAxes,
            fontsize=22,
            fontweight='bold',
            ha='left',
            va='bottom',
            clip_on=False,
        )

    add_single_source_legend(fig, voxel_data, parcel_data)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'png'):
        out_file = output_prefix.with_suffix(f'.{extension}')
        fig.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0.04)
        print(f'Wrote: {out_file}', flush=True)
    plt.close(fig)

    voxel_summary.assign(domain='mni_voxelwise').to_csv(
        output_prefix.with_suffix('.mni_voxelwise_summary.tsv'),
        sep='\t',
        index=False,
    )
    parcel_summary.assign(domain='parcel_bundle').to_csv(
        output_prefix.with_suffix('.parcel_bundle_summary.tsv'),
        sep='\t',
        index=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-set', choices=('primary', 'full'), default='primary')
    parser.add_argument('--stat', choices=('mean', 'median'), default='median')
    parser.add_argument('--mni-icc-dir', type=Path, default=default_mni_icc_dir())
    parser.add_argument(
        '--parcel-bundle-icc-dir', type=Path, default=default_parcel_bundle_icc_dir()
    )
    parser.add_argument(
        '--patterns-file', type=Path, default=CODE_ROOT / 'configuration' / 'metrics.yml'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=RUN_FIGURES_ROOT / 'figure_05_icc',
    )
    parser.add_argument('--output-name', default='Figure5')
    parser.add_argument('--voxelwise-analysis', default='primary')
    parser.add_argument('--max-voxels-per-metric', type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_dependencies(need_nibabel=True)
    output_name = args.output_name
    parcel_data = load_parcel_bundle_icc(
        args.parcel_bundle_icc_dir.expanduser().resolve(),
        args.analysis_set,
        args.stat,
        args.patterns_file.expanduser().resolve(),
    )
    voxel_data = load_voxelwise_icc(
        args.mni_icc_dir.expanduser().resolve(),
        args.analysis_set,
        args.voxelwise_analysis,
        args.patterns_file.expanduser().resolve(),
        args.max_voxels_per_metric,
    )
    plot_combined_icc(
        voxel_data,
        parcel_data,
        args.output_dir.expanduser().resolve() / output_name,
    )


if __name__ == '__main__':
    main()
