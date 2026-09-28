#!/usr/bin/env python3
"""Plot primary combined WM/cortical-GM correlation matrix figures."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.patches import Patch, Rectangle
    from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
    from scipy.spatial.distance import squareform
except ImportError:  # pragma: no cover - checked after argparse handles --help
    mpl = None
    plt = None
    np = None
    pd = None
    sns = None
    GridSpec = None
    GridSpecFromSubplotSpec = None
    Patch = None
    Rectangle = None
    dendrogram = None
    leaves_list = None
    linkage = None
    squareform = None

from utils.metrics import (
    METRIC_FAMILY_LEGEND_TITLE,
    SOURCE_IMAGE_COLORS,
    build_metric_specs,
    metric_display_labels,
)
from utils.paths import CODE_ROOT, OUTPUT_DERIVATIVES_ROOT, RUN_FIGURES_ROOT

from figures._shared.correlation_matrices import (
    PARCEL_CORRELATIONS,
    PARCEL_STAT,
    load_correlation_matrix,
    mni_input_path,
    parcel_input_path,
    selected_values,
    source_display_label,
)


ANALYSIS_SETS = ('primary', 'full')
MNI_CORRELATIONS = ('pearson', 'spearman')
TISSUES = ('wm', 'gm')


@dataclass(frozen=True)
class PanelSpec:
    tissue: str
    title: str
    path: Path
    source_by_label: dict[str, str]


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


def source_lookup(patterns_file: Path, analysis_set: str, tissue: str) -> dict[str, str]:
    specs = build_metric_specs(patterns_file)
    labels = metric_display_labels(specs, analysis_set, tissue=tissue)
    lookup: dict[str, str] = {}
    for spec in specs:
        display = labels.get(spec.label)
        candidates = {spec.label, spec.primary_label, display, labels.get(spec.primary_label)}
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


def ordered_matrix(corr: pd.DataFrame) -> tuple[pd.DataFrame, object | None]:
    z_matrix = correlation_linkage(corr)
    if z_matrix is None:
        return corr.copy(), None
    ordered_labels = [corr.index[index] for index in leaves_list(z_matrix)]
    return corr.loc[ordered_labels, ordered_labels].copy(), z_matrix


def label_fontsize(n_metrics: int) -> float:
    if n_metrics <= 28:
        return 13.0
    if n_metrics <= 45:
        return 10.4
    if n_metrics <= 70:
        return 8.2
    return 6.8


def title_fontsize(n_metrics: int) -> float:
    return max(14.5, min(18.5, 20.0 - 0.09 * n_metrics))


def figure_size(panel_matrices: list[pd.DataFrame]) -> tuple[float, float]:
    max_metrics = max(matrix.shape[0] for matrix in panel_matrices)
    width = max(12.0, min(20.0, 6.2 + 0.39 * max_metrics))
    panel_height = max(5.0, min(11.0, 1.35 + 0.28 * max_metrics))
    return width, 2.0 * panel_height + 2.15


def align_side_axes_to_heatmaps(
    fig: plt.Figure, panel_axes: list[tuple[plt.Axes, plt.Axes, plt.Axes]]
) -> None:
    """Match dendrogram/source-strip boxes to the final square heatmap boxes."""

    fig.canvas.draw()
    for dendro_ax, source_ax, heatmap_ax in panel_axes:
        heatmap_pos = heatmap_ax.get_position()
        dendro_pos = dendro_ax.get_position()
        source_pos = source_ax.get_position()
        source_gap = 0.003
        dendro_gap = 0.002
        source_width = min(source_pos.width, 0.018)
        dendro_width = min(dendro_pos.width, 0.075)
        source_x0 = heatmap_pos.x0 - source_gap - source_width
        dendro_x0 = source_x0 - dendro_gap - dendro_width
        dendro_ax.set_position([dendro_x0, heatmap_pos.y0, dendro_width, heatmap_pos.height])
        source_ax.set_position([source_x0, heatmap_pos.y0, source_width, heatmap_pos.height])


def position_shared_guides(
    fig: plt.Figure,
    cbar_ax: plt.Axes,
    legend_ax: plt.Axes,
    bottom_heatmap_ax: plt.Axes,
) -> None:
    """Place shared guide axes from the actual lower tick-label extent."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_boxes = [
        tick.get_window_extent(renderer)
        for tick in bottom_heatmap_ax.get_xticklabels()
        if tick.get_visible() and tick.get_text()
    ]
    if tick_boxes:
        label_bottom_display = min(box.y0 for box in tick_boxes)
        label_bottom = fig.transFigure.inverted().transform((0, label_bottom_display))[1]
    else:
        label_bottom = bottom_heatmap_ax.get_position().y0

    cbar_width = 0.40
    cbar_height = 0.014
    cbar_y0 = max(0.098, label_bottom - 0.032)
    legend_y0 = max(0.030, cbar_y0 - 0.074)

    cbar_ax.set_position([0.5 - cbar_width / 2.0, cbar_y0, cbar_width, cbar_height])
    legend_ax.set_position([0.12, legend_y0, 0.76, 0.045])


def draw_source_bar(
    ax,
    labels: list[str],
    source_by_label: dict[str, str],
    label_size: float,
    label_rotation: float = 45.0,
    label_fontweight: str = 'normal',
) -> None:
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
    ax.imshow(colors, aspect='auto', interpolation='nearest', origin='upper')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xticks([0])
    ax.set_xticklabels(
        [METRIC_FAMILY_LEGEND_TITLE],
        rotation=label_rotation,
        ha='right',
        rotation_mode='anchor',
        fontweight=label_fontweight,
    )
    ax.tick_params(axis='x', length=0, pad=7, labelsize=label_size)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_left_dendrogram(ax, z_matrix, n_metrics: int) -> None:
    ax.set_axis_off()
    if z_matrix is None:
        return
    dendrogram(
        z_matrix,
        orientation='left',
        no_labels=True,
        color_threshold=0,
        above_threshold_color='#2b2b2b',
        link_color_func=lambda _: '#2b2b2b',
        ax=ax,
    )
    ax.invert_yaxis()
    ax.set_ylim(n_metrics * 10, 0)
    for collection in ax.collections:
        collection.set_linewidth(1.25)
        collection.set_color('#2b2b2b')


def draw_heatmap(
    ax,
    corr: pd.DataFrame,
    cmap,
    label_size: float,
) -> None:
    plot_data = corr.copy()
    np.fill_diagonal(plot_data.values, np.nan)
    masked = np.ma.masked_invalid(plot_data.to_numpy(dtype=float))
    ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1, interpolation='nearest', origin='upper')

    n_metrics = corr.shape[0]
    for index in range(n_metrics):
        ax.add_patch(
            Rectangle(
                (index - 0.5, index - 0.5),
                1,
                1,
                facecolor='black',
                edgecolor='black',
                linewidth=0,
                zorder=4,
            )
        )

    ticks = np.arange(n_metrics)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(list(corr.columns), rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(list(corr.index))
    ax.yaxis.tick_right()
    ax.tick_params(axis='both', length=0, pad=4, labelsize=label_size)
    ax.set_xlim(-0.5, n_metrics - 0.5)
    ax.set_ylim(n_metrics - 0.5, -0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('')
    ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_visible(False)


def observed_sources(panel_matrices: list[pd.DataFrame], panel_specs: list[PanelSpec]) -> list[str]:
    observed: set[str] = set()
    for matrix, spec in zip(panel_matrices, panel_specs, strict=True):
        observed.update(spec.source_by_label.get(label, 'Other') for label in matrix.index)
    return [source for source in SOURCE_IMAGE_COLORS if source in observed]


def cbar_label(kind: str, correlation: str) -> str:
    method = r'Pearson $r$' if correlation == 'pearson' else 'Spearman ρ'
    if kind == 'mni':
        return f'Mean voxelwise {method}'
    return f'Mean {method}'


def draw_combined_figure(
    panels: list[PanelSpec],
    kind: str,
    analysis_set: str,
    correlation: str,
    out_stem: Path,
) -> None:
    loaded = [load_correlation_matrix(panel.path) for panel in panels]
    ordered = [ordered_matrix(matrix) for matrix in loaded]
    matrices = [item[0] for item in ordered]
    linkages = [item[1] for item in ordered]
    max_metrics = max(matrix.shape[0] for matrix in matrices)
    fs = label_fontsize(max_metrics)

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    cmap.set_bad('#eeeeee')

    fig_width, fig_height = figure_size(matrices)
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    outer = GridSpec(
        4,
        1,
        figure=fig,
        height_ratios=[1, 1, 0.012, 0.025],
        hspace=0.62,
    )

    image = None
    panel_axes: list[tuple[plt.Axes, plt.Axes, plt.Axes]] = []
    for panel_index, (panel, matrix, z_matrix) in enumerate(
        zip(panels, matrices, linkages, strict=True)
    ):
        inner = GridSpecFromSubplotSpec(
            1,
            3,
            subplot_spec=outer[panel_index],
            width_ratios=[0.078, 0.020, 0.902],
            wspace=0.004,
        )
        dendro_ax = fig.add_subplot(inner[0, 0])
        source_ax = fig.add_subplot(inner[0, 1])
        heatmap_ax = fig.add_subplot(inner[0, 2])

        draw_left_dendrogram(dendro_ax, z_matrix, matrix.shape[0])
        draw_source_bar(source_ax, list(matrix.index), panel.source_by_label, max(9.0, fs - 0.4))
        draw_heatmap(heatmap_ax, matrix, cmap, fs)
        image = heatmap_ax.images[0]
        panel_axes.append((dendro_ax, source_ax, heatmap_ax))

        heatmap_ax.set_title(
            panel.title,
            loc='left',
            fontsize=title_fontsize(max_metrics),
            fontweight='bold',
            pad=7,
        )
        heatmap_ax.text(
            -0.15,
            1.02,
            chr(ord('A') + panel_index),
            transform=heatmap_ax.transAxes,
            ha='left',
            va='bottom',
            fontsize=title_fontsize(max_metrics) + 1.5,
            fontweight='bold',
            clip_on=False,
        )

    fig.subplots_adjust(left=0.035, right=0.955, top=0.982, bottom=0.045)
    align_side_axes_to_heatmaps(fig, panel_axes)

    cbar_ax = fig.add_subplot(outer[2])
    if image is None:
        raise RuntimeError('No matrices were plotted.')
    cbar = fig.colorbar(image, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=max(10.0, fs - 0.3), length=3)
    cbar.set_label(cbar_label(kind, correlation), fontsize=max(11.0, fs + 0.2), labelpad=5)
    cbar.ax.xaxis.label.set_fontweight('bold')

    legend_ax = fig.add_subplot(outer[3])
    legend_ax.axis('off')
    sources = observed_sources(matrices, panels)
    handles = [
        Patch(
            facecolor=SOURCE_IMAGE_COLORS[source],
            edgecolor='none',
            label=source_display_label(source),
        )
        for source in sources
    ]
    legend = legend_ax.legend(
        handles=handles,
        title=METRIC_FAMILY_LEGEND_TITLE,
        loc='center',
        ncol=max(1, len(handles)),
        frameon=False,
        fontsize=max(10.5, fs),
        title_fontsize=max(11.5, fs + 0.5),
        handlelength=1.5,
        columnspacing=1.6,
    )
    legend.get_title().set_fontweight('bold')
    position_shared_guides(fig, cbar_ax, legend_ax, panel_axes[-1][2])

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'png'):
        out_path = out_stem.with_suffix(f'.{extension}')
        fig.savefig(out_path, bbox_inches='tight', dpi=300)
        print(f'Wrote: {out_path}', flush=True)
    plt.close(fig)


def build_panel_specs(
    kind: str,
    analysis_set: str,
    correlation: str,
    patterns_file: Path,
    mni_dir: Path,
    parcel_dir: Path,
    parcel_stat: str,
) -> list[PanelSpec]:
    if kind == 'mni':
        return [
            PanelSpec(
                tissue='wm',
                title='White Matter Voxels',
                path=mni_input_path(mni_dir, analysis_set, 'wm', correlation),
                source_by_label=source_lookup(patterns_file, analysis_set, 'wm'),
            ),
            PanelSpec(
                tissue='gm',
                title='Cortical Gray Matter Voxels',
                # plot-facing "gm" resolves to the cortical_gm voxel matrix.
                path=mni_input_path(mni_dir, analysis_set, 'gm', correlation),
                source_by_label=source_lookup(patterns_file, analysis_set, 'gm'),
            ),
        ]
    return [
        PanelSpec(
            tissue='wm',
            title='White Matter Bundles',
            path=parcel_input_path(parcel_dir, analysis_set, 'wm', correlation, parcel_stat),
            source_by_label=source_lookup(patterns_file, analysis_set, 'wm'),
        ),
        PanelSpec(
            tissue='gm',
            title='Gray Matter Parcels',
            path=parcel_input_path(parcel_dir, analysis_set, 'gm', correlation, parcel_stat),
            source_by_label=source_lookup(patterns_file, analysis_set, 'gm'),
        ),
    ]


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
        default=RUN_FIGURES_ROOT / 'correlation_matrices_combined',
        help='Directory for combined correlation matrix figures.',
    )
    parser.add_argument(
        '--analysis-set',
        nargs='+',
        choices=('primary', 'full', 'both'),
        default=['primary'],
        help='Metric set(s) to plot.',
    )
    parser.add_argument(
        '--kind',
        nargs='+',
        choices=('mni', 'parcel', 'both'),
        default=['both'],
        help='Combined figure type(s): MNI voxelwise, parcel/bundle, or both.',
    )
    parser.add_argument(
        '--mni-correlation',
        nargs='+',
        choices=('pearson', 'spearman', 'both'),
        default=['pearson'],
        help='MNI voxelwise correlation method(s) to plot.',
    )
    parser.add_argument(
        '--parcel-correlation',
        nargs='+',
        choices=(*PARCEL_CORRELATIONS, 'both'),
        default=['pearson'],
        help='Parcel/bundle correlation method(s) to plot.',
    )
    parser.add_argument(
        '--parcel-stat',
        choices=('mean', 'median'),
        default=PARCEL_STAT,
        help='Parcel/bundle summary statistic used in parcel correlation filenames.',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Fail instead of warning when an expected input matrix is missing.',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_dependencies()
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    sns.set_theme(style='white', font='Arial')

    analysis_sets = selected_values(args.analysis_set, ANALYSIS_SETS)
    kinds = selected_values(args.kind, ('mni', 'parcel'))
    mni_correlations = selected_values(args.mni_correlation, MNI_CORRELATIONS)
    parcel_correlations = selected_values(args.parcel_correlation, PARCEL_CORRELATIONS)

    for analysis_set in analysis_sets:
        for kind in kinds:
            correlations = mni_correlations if kind == 'mni' else parcel_correlations
            for correlation in correlations:
                panels = build_panel_specs(
                    kind,
                    analysis_set,
                    correlation,
                    args.patterns_file,
                    args.mni_dir,
                    args.parcel_dir,
                    args.parcel_stat,
                )
                missing = [panel.path for panel in panels if not panel.path.exists()]
                if missing:
                    message = 'Missing input(s), skipping combined figure: ' + ', '.join(
                        map(str, missing)
                    )
                    if args.strict:
                        raise FileNotFoundError(message)
                    print(f'[WARN] {message}', flush=True)
                    continue
                stat_token = f'_{args.parcel_stat}' if kind == 'parcel' else ''
                out_stem = (
                    args.output_dir
                    / f'combined_{kind}_{analysis_set}_{correlation}{stat_token}_correlations'
                )
                draw_combined_figure(panels, kind, analysis_set, correlation, out_stem)


if __name__ == '__main__':
    main()
