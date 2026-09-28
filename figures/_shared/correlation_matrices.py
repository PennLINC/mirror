#!/usr/bin/env python3
"""Plot clustered metric correlation matrices from saved correlation TSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib.patches import Patch, Rectangle
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
except ImportError:  # pragma: no cover - checked after argparse handles --help
    mpl = None
    plt = None
    np = None
    pd = None
    sns = None
    Patch = None
    Rectangle = None
    linkage = None
    squareform = None

from utils.metrics import (
    METRIC_FAMILY_LEGEND_TITLE,
    SOURCE_IMAGE_COLORS,
    build_metric_specs,
    metric_display_labels,
    metric_plot_label,
    source_image_display_label,
)
from utils.paths import CODE_ROOT, OUTPUT_DERIVATIVES_ROOT, RUN_FIGURES_ROOT


ANALYSIS_SETS = ('primary', 'full')
TISSUES = ('gm', 'wm')
MNI_CORRELATIONS = ('pearson', 'spearman')
PARCEL_STAT = 'median'
PARCEL_CORRELATIONS = ('spearman', 'pearson')


def require_dependencies() -> None:
    missing = [
        name
        for name, module in (
            ('matplotlib', mpl),
            ('numpy', np),
            ('pandas', pd),
            ('seaborn', sns),
            ('scipy.cluster', linkage),
        )
        if module is None
    ]
    if missing:
        raise RuntimeError(
            'Missing required Python packages: '
            f'{", ".join(missing)}. Activate the MIRROR analysis environment first.'
        )


def default_parcel_dir() -> Path:
    return OUTPUT_DERIVATIVES_ROOT / 'parcel_bundle_correlations'


def clean_title_token(value: str) -> str:
    return str(value).replace('_', ' ').title()


def figure_size(n_metrics: int) -> tuple[float, float]:
    # Very large canvases are aggressively downscaled when placed in a paper,
    # which makes their text much smaller than its nominal point size.
    side = max(10.5, min(20.0, 5.0 + 0.18 * n_metrics))
    return side, side


def label_fontsize(n_metrics: int) -> float:
    if n_metrics <= 28:
        return 15.0
    if n_metrics <= 45:
        return 14.5
    if n_metrics <= 70:
        return 14.0
    return 10


def title_fontsize(n_metrics: int) -> float:
    return max(21.0, min(27.0, 30.0 - 0.09 * n_metrics))


def source_display_label(source: str) -> str:
    return source_image_display_label(source)


def load_correlation_matrix(path: Path) -> pd.DataFrame:
    corr = pd.read_csv(path, sep='\t', index_col=0)
    corr.index = corr.index.astype(str)
    corr.columns = corr.columns.astype(str)
    labels = list(corr.index)
    missing_columns = [label for label in labels if label not in corr.columns]
    if missing_columns:
        raise RuntimeError(
            f'{path} is not a square metric-by-metric matrix; missing columns: '
            f'{", ".join(missing_columns[:10])}'
        )
    corr = corr.loc[labels, labels].apply(pd.to_numeric, errors='coerce')
    values = corr.to_numpy(dtype=float)
    values = (values + values.T) / 2.0
    corr.loc[:, :] = values
    display_labels = [metric_plot_label(label) for label in corr.index]
    if len(set(display_labels)) != len(display_labels):
        raise RuntimeError(
            f'{path} contains metric labels that collide after publication formatting.'
        )
    corr.index = display_labels
    corr.columns = [metric_plot_label(label) for label in corr.columns]
    return corr


def source_lookup(patterns_file: Path, analysis_set: str, tissue: str) -> dict[str, str]:
    specs = build_metric_specs(patterns_file)
    labels = metric_display_labels(specs, analysis_set, tissue=tissue)
    lookup: dict[str, str] = {}
    for spec in specs:
        display = labels.get(spec.label)
        candidates = {
            spec.label,
            spec.primary_label,
            display,
            labels.get(spec.primary_label),
        }
        for candidate in candidates:
            if candidate:
                lookup[str(candidate)] = spec.source_image
    return lookup


def correlation_linkage(corr: pd.DataFrame):
    if corr.shape[0] < 2:
        return None
    safe = corr.fillna(0.0).to_numpy(dtype=float)
    safe = np.clip((safe + safe.T) / 2.0, -1.0, 1.0)
    distance = np.clip(1.0 - np.abs(safe), 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    return linkage(squareform(distance, checks=False), method='average', optimal_ordering=True)


def draw_diagonal(grid, color: str = 'black') -> None:
    row_order = list(grid.data2d.index)
    column_order = list(grid.data2d.columns)
    column_position = {label: index for index, label in enumerate(column_order)}
    for row_index, label in enumerate(row_order):
        col_index = column_position.get(label)
        if col_index is None:
            continue
        grid.ax_heatmap.add_patch(
            Rectangle(
                (col_index, row_index),
                1,
                1,
                facecolor=color,
                edgecolor=color,
                linewidth=0,
                zorder=5,
            )
        )


def add_source_annotation(
    grid,
    source_by_label: dict[str, str],
    width: float = 0.020,
    gap: float = 0.001,
) -> None:
    grid.fig.canvas.draw()
    heatmap_position = grid.ax_heatmap.get_position()
    dendrogram_position = grid.ax_row_dendrogram.get_position()
    bar_x0 = heatmap_position.x0 - width - gap
    dendrogram_gap = 0.002
    dendrogram_width = min(
        dendrogram_position.width,
        max(0.040, bar_x0 - dendrogram_position.x0 - dendrogram_gap),
    )
    grid.ax_row_dendrogram.set_position(
        [
            bar_x0 - dendrogram_gap - dendrogram_width,
            heatmap_position.y0,
            dendrogram_width,
            heatmap_position.height,
        ]
    )

    labels = list(grid.data2d.index)
    colors = np.array(
        [
            mpl.colors.to_rgba(
                SOURCE_IMAGE_COLORS.get(
                    source_by_label.get(label, 'Other'),
                    SOURCE_IMAGE_COLORS['Other'],
                )
            )
            for label in labels
        ]
    ).reshape(len(labels), 1, 4)
    bar_ax = grid.fig.add_axes(
        [
            bar_x0,
            heatmap_position.y0,
            width,
            heatmap_position.height,
        ]
    )
    bar_ax.imshow(colors, aspect='auto', interpolation='nearest', origin='upper')
    bar_ax.set_xlim(-0.5, 0.5)
    bar_ax.set_ylim(len(labels) - 0.5, -0.5)
    bar_ax.set_xticks([])
    bar_ax.set_yticks([])
    for spine in bar_ax.spines.values():
        spine.set_visible(False)


def style_row_dendrogram(grid, linewidth: float = 2.8) -> None:
    """Make hierarchical-clustering branches legible in full-metric figures."""

    for collection in grid.ax_row_dendrogram.collections:
        collection.set_linewidth(linewidth)
        collection.set_color('#2b2b2b')


def position_matrix_guides(grid, legend) -> None:
    """Place the colorbar and legend directly below the rotated metric labels."""

    grid.fig.canvas.draw()
    renderer = grid.fig.canvas.get_renderer()
    tick_boxes = [
        tick.get_window_extent(renderer)
        for tick in grid.ax_heatmap.get_xticklabels()
        if tick.get_visible() and tick.get_text()
    ]
    if tick_boxes:
        label_bottom_display = min(box.y0 for box in tick_boxes)
        label_bottom = grid.fig.transFigure.inverted().transform((0, label_bottom_display))[1]
    else:
        label_bottom = grid.ax_heatmap.get_position().y0

    heatmap_position = grid.ax_heatmap.get_position()
    cbar_width = min(0.42, 0.62 * heatmap_position.width)
    cbar_x0 = heatmap_position.x0 + 0.5 * (heatmap_position.width - cbar_width)
    cbar_y0 = max(0.050, label_bottom - 0.028)
    grid.cax.set_position([cbar_x0, cbar_y0, cbar_width, 0.018])
    legend.set_bbox_to_anchor(
        (heatmap_position.x0 + 0.5 * heatmap_position.width, cbar_y0 - 0.030),
        transform=grid.fig.transFigure,
    )


def plot_matrix(
    corr: pd.DataFrame,
    source_by_label: dict[str, str],
    title: str,
    cbar_label: str,
    out_stem: Path,
) -> None:
    n_metrics = corr.shape[0]
    z_matrix = correlation_linkage(corr)
    plot_data = corr.copy()
    np.fill_diagonal(plot_data.values, np.nan)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    cmap.set_bad('#eeeeee')

    grid = sns.clustermap(
        plot_data,
        row_linkage=z_matrix,
        col_linkage=z_matrix,
        row_cluster=z_matrix is not None,
        col_cluster=z_matrix is not None,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0,
        xticklabels=True,
        yticklabels=True,
        figsize=figure_size(n_metrics),
        dendrogram_ratio=(0.075, 0.001),
        cbar_pos=(0.29, 0.065, 0.42, 0.022),
        cbar_kws={
            'orientation': 'horizontal',
            'label': cbar_label,
            'ticks': [-1, -0.5, 0, 0.5, 1],
        },
    )
    grid.ax_col_dendrogram.set_visible(False)
    style_row_dendrogram(grid)
    grid.ax_heatmap.set_aspect('equal', adjustable='box')
    grid.ax_heatmap.set_xlabel('')
    grid.ax_heatmap.set_ylabel('')
    grid.ax_heatmap.tick_params(axis='both', length=0, pad=4)

    fs = label_fontsize(n_metrics)
    ticks = np.arange(n_metrics) + 0.5
    grid.ax_heatmap.set_xticks(ticks)
    grid.ax_heatmap.set_yticks(ticks)
    grid.ax_heatmap.set_xticklabels(list(grid.data2d.columns))
    grid.ax_heatmap.set_yticklabels(list(grid.data2d.index))
    plt.setp(
        grid.ax_heatmap.get_xticklabels(),
        rotation=45,
        ha='right',
        rotation_mode='anchor',
        fontsize=fs,
    )
    plt.setp(
        grid.ax_heatmap.get_yticklabels(),
        rotation=0,
        fontsize=fs,
    )

    observed_sources = [
        source
        for source in SOURCE_IMAGE_COLORS
        if source in {source_by_label.get(label, 'Other') for label in plot_data.index}
    ]
    handles = [
        Patch(
            facecolor=SOURCE_IMAGE_COLORS[source],
            edgecolor='none',
            label=source_display_label(source),
        )
        for source in observed_sources
    ]
    legend = grid.fig.legend(
        handles=handles,
        title=METRIC_FAMILY_LEGEND_TITLE,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.025),
        ncol=max(1, len(handles)),
        frameon=False,
        fontsize=max(13.0, fs - 0.2),
        title_fontsize=max(14.0, fs + 0.5),
        handlelength=1.5,
        columnspacing=1.25,
    )
    legend.get_title().set_fontweight('bold')

    grid.fig.suptitle(
        title,
        fontsize=title_fontsize(n_metrics),
        fontweight='bold',
        y=0.975,
    )
    grid.fig.subplots_adjust(left=0.052, right=0.93, top=0.953, bottom=0.180)
    grid.cax.tick_params(labelsize=max(13.0, fs - 0.2), length=4)
    grid.cax.xaxis.set_label_position('top')
    grid.cax.xaxis.label.set_size(max(14.0, fs + 0.5))
    grid.cax.xaxis.label.set_fontweight('bold')
    grid.cax.xaxis.labelpad = 5
    draw_diagonal(grid)
    add_source_annotation(grid, source_by_label)
    position_matrix_guides(grid, legend)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'png'):
        out_path = out_stem.with_suffix(f'.{extension}')
        grid.fig.savefig(out_path, bbox_inches='tight', dpi=300)
        print(f'Wrote: {out_path}', flush=True)
    plt.close(grid.fig)


def mni_input_path(mni_dir: Path, analysis_set: str, tissue: str, correlation: str) -> Path:
    mni_tissue = 'cortical_gm' if tissue == 'gm' else tissue
    return mni_dir / f'mni_voxelwise_{analysis_set}_{mni_tissue}_{correlation}_r.tsv'


def parcel_input_path(
    parcel_dir: Path,
    analysis_set: str,
    tissue: str,
    correlation: str,
    stat: str,
) -> Path:
    profile_type = 'gm_parcels' if tissue == 'gm' else 'wm_bundles'
    return parcel_dir / f'{profile_type}_{analysis_set}_{correlation}_{stat}_r.tsv'


def selected_values(values: list[str], all_values: tuple[str, ...]) -> list[str]:
    if 'both' in values:
        return list(all_values)
    return values


def plot_if_available(
    input_path: Path,
    output_stem: Path,
    source_by_label: dict[str, str],
    title: str,
    cbar_label: str,
    strict: bool,
) -> None:
    if not input_path.exists():
        message = f'Missing input, skipping: {input_path}'
        if strict:
            raise FileNotFoundError(message)
        print(f'[WARN] {message}', flush=True)
        return
    corr = load_correlation_matrix(input_path)
    if corr.empty:
        print(f'[WARN] Empty matrix, skipping: {input_path}', flush=True)
        return
    plot_matrix(corr, source_by_label, title, cbar_label, output_stem)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--patterns-file',
        type=Path,
        default=CODE_ROOT / 'configuration' / 'metrics.yml',
        help='Metric pattern registry used to assign source-image colors.',
    )
    parser.add_argument(
        '--mni-dir',
        type=Path,
        default=OUTPUT_DERIVATIVES_ROOT / 'mni_voxelwise_correlations',
        help='Directory containing compute_mni_voxelwise_correlations.py TSV outputs.',
    )
    parser.add_argument(
        '--parcel-dir',
        type=Path,
        default=default_parcel_dir(),
        help='Directory containing compute_parcel_bundle_correlations.py TSV outputs.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=RUN_FIGURES_ROOT / 'correlation_matrices',
        help='Directory for clustered correlation matrix figures.',
    )
    parser.add_argument(
        '--output-name',
        default=None,
        help='Exact output stem for a single requested matrix.',
    )
    parser.add_argument(
        '--analysis-set',
        nargs='+',
        choices=('primary', 'full', 'both'),
        default=['both'],
        help='Metric set(s) to plot.',
    )
    parser.add_argument(
        '--tissue',
        nargs='+',
        choices=('gm', 'wm', 'both'),
        default=['both'],
        help='Tissue/profile(s) to plot.',
    )
    parser.add_argument(
        '--mni-correlation',
        nargs='+',
        choices=('pearson', 'spearman', 'both'),
        default=['pearson'],
        help='MNI voxelwise correlation method(s) to plot.',
    )
    parser.add_argument(
        '--parcel-stat',
        choices=('mean', 'median'),
        default=PARCEL_STAT,
        help='Parcel/bundle summary statistic used in parcel correlation filenames.',
    )
    parser.add_argument(
        '--parcel-correlation',
        nargs='+',
        choices=(*PARCEL_CORRELATIONS, 'both'),
        default=['both'],
        help='GM parcel / WM bundle correlation method(s) to plot.',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Fail instead of warning when an expected input matrix is missing.',
    )
    parser.add_argument(
        '--skip-mni',
        action='store_true',
        help='Do not plot MNI voxelwise matrices.',
    )
    parser.add_argument(
        '--skip-parcel',
        action='store_true',
        help='Do not plot GM parcel / WM bundle matrices.',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Render requested correlation matrices."""

    args = build_parser().parse_args(argv)
    require_dependencies()
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    sns.set_theme(style='white', font='Arial')

    analysis_sets = selected_values(args.analysis_set, ANALYSIS_SETS)
    tissues = selected_values(args.tissue, TISSUES)
    mni_correlations = selected_values(args.mni_correlation, MNI_CORRELATIONS)
    parcel_correlations = selected_values(args.parcel_correlation, PARCEL_CORRELATIONS)

    for analysis_set in analysis_sets:
        for tissue in tissues:
            source_by_label = source_lookup(args.patterns_file, analysis_set, tissue)
            tissue_title = 'Cortical GM' if tissue == 'gm' else 'WM'

            if not args.skip_mni:
                for correlation in mni_correlations:
                    corr_title = 'Pearson' if correlation == 'pearson' else 'Spearman'
                    title = (
                        f'{clean_title_token(analysis_set)} {tissue_title} '
                        f'MNI Voxelwise {corr_title} Correlations'
                    )
                    output_name = args.output_name or (
                        f'mni_voxelwise_{analysis_set}_{tissue}_{correlation}_correlations'
                    )
                    out_stem = args.output_dir / output_name
                    plot_if_available(
                        mni_input_path(args.mni_dir, analysis_set, tissue, correlation),
                        out_stem,
                        source_by_label,
                        title,
                        r'Mean voxelwise Pearson $r$'
                        if correlation == 'pearson'
                        else 'Mean voxelwise Spearman ρ',
                        args.strict,
                    )

            if not args.skip_parcel:
                profile_title = 'GM Parcels' if tissue == 'gm' else 'WM Bundles'
                for correlation in parcel_correlations:
                    corr_title = 'Pearson' if correlation == 'pearson' else 'Spearman'
                    title = (
                        f'{clean_title_token(analysis_set)} {profile_title} '
                        f'{corr_title} Correlations'
                    )
                    output_name = args.output_name or (
                        f'parcel_bundle_{analysis_set}_{tissue}_{correlation}_'
                        f'{args.parcel_stat}_correlations'
                    )
                    out_stem = args.output_dir / output_name
                    plot_if_available(
                        parcel_input_path(
                            args.parcel_dir,
                            analysis_set,
                            tissue,
                            correlation,
                            args.parcel_stat,
                        ),
                        out_stem,
                        source_by_label,
                        title,
                        r'Mean Pearson $r$' if correlation == 'pearson' else 'Mean Spearman ρ',
                        args.strict,
                    )


if __name__ == '__main__':
    main()
