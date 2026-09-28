#!/usr/bin/env python3
"""Plot full-metric regional ICC heatmaps and voxelwise ICC interval figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Patch, Rectangle
except ImportError:  # pragma: no cover
    mpl = None
    plt = None
    np = None
    pd = None
    Patch = None
    Rectangle = None

from utils.metrics import (
    METRIC_FAMILY_LEGEND_TITLE,
    SOURCE_IMAGE_COLORS,
    build_metric_specs,
    metric_display_labels,
    source_image_display_label,
)
from utils.paths import CODE_ROOT, RUN_FIGURES_ROOT
from utils.regional_io import (
    EXPECTED_WM_BUNDLE_COUNT,
    canonical_wm_bundle_name,
    expected_wm_bundle_ids,
)
from utils.regions import regional_feature_display_label

from figures._shared.icc import (
    BENCHMARKS,
    default_mni_icc_dir,
    default_parcel_bundle_icc_dir,
    load_voxelwise_icc,
    require_dependencies,
    summarize_metric_values,
)


REGIONAL_DOMAINS = {
    'wm': ('wm_bundles', 'White Matter Bundles'),
    'gm': ('gm_parcels', 'Cortical Gray Matter Parcels'),
}
VOXEL_TITLES = {
    'wm': 'White Matter Voxelwise ICC',
    'gm': 'Cortical Gray Matter Voxelwise ICC',
}


def icc_display_label(icc_column: str) -> str:
    return {'ICC2_1': 'ICC(2,1)', 'ICC3_1': 'ICC(3,1)'}[icc_column]


def regional_input_path(icc_dir: Path, tissue: str, analysis_set: str, stat: str) -> Path:
    profile, _ = REGIONAL_DOMAINS[tissue]
    return icc_dir / f'icc_{profile}_{analysis_set}_{stat}.csv'


def load_regional_icc(
    path: Path,
    tissue: str,
    patterns_file: Path,
    analysis_set: str,
    icc_column: str,
) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {'feature', icc_column}
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError(f'{path} is missing required columns: {", ".join(sorted(missing))}')
    if 'metric_key' not in table:
        if 'metric' not in table:
            raise RuntimeError(f'{path} has neither metric_key nor metric.')
        table['metric_key'] = table['metric'].astype(str)

    specs = build_metric_specs(patterns_file)
    displays = metric_display_labels(specs, analysis_set, tissue=tissue)
    sources = {spec.label: spec.source_image for spec in specs}
    table = table.copy()
    table[icc_column] = pd.to_numeric(table[icc_column], errors='coerce')
    table['metric_key'] = table['metric_key'].astype(str)
    if tissue == 'wm':
        raw_features = sorted(table['feature'].astype(str).unique())
        feature_map = {feature: canonical_wm_bundle_name(feature) for feature in raw_features}
        unmatched = sorted(
            feature for feature, canonical in feature_map.items() if canonical is None
        )
        if unmatched:
            raise RuntimeError(
                f'{path} contains {len(unmatched)} bundle names not found in '
                f'processing/01_smri_dmri/qsirecon_spec.yml: {", ".join(unmatched[:20])}'
            )
        table['feature'] = table['feature'].astype(str).map(feature_map)
        print(
            '[INFO] Canonicalized regional ICC bundle names: '
            f'{len(raw_features)} raw names -> {table["feature"].nunique()} AutoTrack IDs.',
            flush=True,
        )
    table['metric'] = table['metric_key'].map(displays).fillna(table['metric_key'])
    table['source_image'] = (
        table['metric_key'].map(sources).fillna(table.get('source_image', 'Other'))
    )
    table = table.dropna(subset=[icc_column]).copy()
    duplicate = table.duplicated(['metric_key', 'feature'], keep=False)
    if duplicate.any():
        examples = (
            table.loc[duplicate, ['metric_key', 'feature']]
            .drop_duplicates()
            .head(10)
            .apply(lambda row: f'{row.metric_key}/{row.feature}', axis=1)
            .tolist()
        )
        raise RuntimeError(
            'Canonicalization found multiple ICC estimates for the same metric/bundle. '
            'Rerun compute_parcel_bundle_icc.py from the canonicalized scalarstats instead '
            f'of averaging ICC estimates. Examples: {", ".join(examples)}'
        )
    return table


def ordered_icc_matrix(
    table: pd.DataFrame,
    icc_column: str,
    expected_features: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, str], dict[str, str]]:
    matrix = table.pivot_table(
        index='feature',
        columns='metric_key',
        values=icc_column,
        aggfunc='mean',
    )
    if expected_features is not None:
        unexpected = sorted(set(matrix.index) - set(expected_features))
        if unexpected:
            raise RuntimeError(
                'Regional ICC table contains unexpected features after canonicalization: '
                + ', '.join(unexpected[:20])
            )
        matrix = matrix.reindex(expected_features)
    row_means = matrix.mean(axis=1, skipna=True).sort_values(ascending=False)
    column_means = matrix.mean(axis=0, skipna=True).sort_values(ascending=False)
    matrix = matrix.loc[row_means.index, column_means.index]
    display = (
        table.drop_duplicates('metric_key').set_index('metric_key')['metric'].astype(str).to_dict()
    )
    source = (
        table.drop_duplicates('metric_key')
        .set_index('metric_key')['source_image']
        .astype(str)
        .to_dict()
    )
    return matrix, row_means, column_means, display, source


def save_order_table(
    output_stem: Path,
    row_means: pd.Series,
    column_means: pd.Series,
    display: dict[str, str],
) -> None:
    records = []
    for rank, (feature, value) in enumerate(row_means.items(), start=1):
        records.append(
            {'axis': 'region', 'rank': rank, 'key': feature, 'label': feature, 'mean_icc': value}
        )
    for rank, (metric, value) in enumerate(column_means.items(), start=1):
        records.append(
            {
                'axis': 'metric',
                'rank': rank,
                'key': metric,
                'label': display.get(metric, metric),
                'mean_icc': value,
            }
        )
    pd.DataFrame(records).to_csv(
        output_stem.with_name(output_stem.name + '_ordering.tsv'),
        sep='\t',
        index=False,
    )


def save_coverage_table(
    output_stem: Path,
    matrix: pd.DataFrame,
    display: dict[str, str],
) -> None:
    records: list[dict[str, object]] = []
    for feature, values in matrix.iterrows():
        available = int(values.notna().sum())
        records.append(
            {
                'axis': 'region',
                'key': feature,
                'label': feature,
                'n_available': available,
                'n_missing': int(len(values) - available),
                'available_fraction': available / len(values) if len(values) else np.nan,
            }
        )
    for metric in matrix.columns:
        values = matrix[metric]
        available = int(values.notna().sum())
        records.append(
            {
                'axis': 'metric',
                'key': metric,
                'label': display.get(metric, metric),
                'n_available': available,
                'n_missing': int(len(values) - available),
                'available_fraction': available / len(values) if len(values) else np.nan,
            }
        )
    pd.DataFrame(records).to_csv(
        output_stem.with_name(output_stem.name + '_coverage.tsv'),
        sep='\t',
        index=False,
    )


def position_regional_guides(
    fig: plt.Figure,
    heatmap_ax: plt.Axes,
    cbar_ax: plt.Axes,
    legend,
) -> None:
    """Place the colorbar and legend tightly below the rotated metric labels."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_boxes = [
        tick.get_window_extent(renderer)
        for tick in heatmap_ax.get_xticklabels()
        if tick.get_visible() and tick.get_text()
    ]
    if heatmap_ax.xaxis.label.get_visible() and heatmap_ax.xaxis.label.get_text():
        tick_boxes.append(heatmap_ax.xaxis.label.get_window_extent(renderer))
    if tick_boxes:
        label_bottom_display = min(box.y0 for box in tick_boxes)
        label_bottom = fig.transFigure.inverted().transform((0, label_bottom_display))[1]
    else:
        label_bottom = heatmap_ax.get_position().y0

    cbar_width = min(0.40, 0.62 * heatmap_ax.get_position().width)
    cbar_x0 = heatmap_ax.get_position().x0 + 0.5 * (heatmap_ax.get_position().width - cbar_width)
    cbar_y0 = max(0.052, label_bottom - 0.030)
    cbar_ax.set_position([cbar_x0, cbar_y0, cbar_width, 0.015])
    legend.set_bbox_to_anchor((0.5, cbar_y0 - 0.030), transform=fig.transFigure)


def plot_regional_heatmap(
    table: pd.DataFrame,
    tissue: str,
    icc_column: str,
    icc_label: str,
    output_stem: Path,
) -> None:
    expected_features = expected_wm_bundle_ids() if tissue == 'wm' else None
    matrix, row_means, column_means, display, source = ordered_icc_matrix(
        table,
        icc_column,
        expected_features=expected_features,
    )
    if matrix.empty:
        raise RuntimeError(f'No finite regional ICC values for {tissue}.')

    n_regions, n_metrics = matrix.shape
    if tissue == 'wm' and n_regions != EXPECTED_WM_BUNDLE_COUNT:
        raise RuntimeError(
            f'WM ICC heatmap must contain {EXPECTED_WM_BUNDLE_COUNT} bundles; got {n_regions}.'
        )
    missing_cells = int(matrix.isna().to_numpy().sum())
    total_cells = int(matrix.size)
    print(
        f'[INFO] {REGIONAL_DOMAINS[tissue][1]} ICC matrix: {n_metrics} metric rows x '
        f'{n_regions} region columns; {missing_cells}/{total_cells} cells missing '
        f'({missing_cells / total_cells:.1%}).',
        flush=True,
    )
    # Keep the source canvas compact enough that manuscript downscaling does
    # not turn otherwise reasonable point sizes into illegible text.
    fig_width = max(15.0, min(20.0, 5.2 + 0.18 * n_regions))
    fig_height = max(12.0, min(26.0, 4.5 + 0.23 * n_metrics))
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.018, 1.0],
        left=0.22,
        right=0.975,
        bottom=0.22,
        top=0.925,
        wspace=0.004,
    )
    family_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1], sharey=family_ax)

    cmap = mpl.colormaps['viridis'].copy()
    cmap.set_bad('#eeeeee')
    cmap.set_under('#52245f')
    image = ax.imshow(
        matrix.T.to_numpy(dtype=float),
        aspect='auto',
        interpolation='nearest',
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    metric_sources = [source.get(metric, 'Other') for metric in matrix.columns]
    family_colors = np.asarray(
        [
            mpl.colors.to_rgba(SOURCE_IMAGE_COLORS.get(item, SOURCE_IMAGE_COLORS['Other']))
            for item in metric_sources
        ]
    ).reshape(n_metrics, 1, 4)
    family_ax.imshow(family_colors, aspect='auto', interpolation='nearest')
    family_ax.set_xticks([])
    family_ax.set_yticks(np.arange(n_metrics))
    family_ax.set_yticklabels(
        [display.get(metric, metric) for metric in matrix.columns],
        fontsize=max(10.2, min(12.5, 1000.0 / max(n_metrics, 1))),
    )
    family_ax.tick_params(axis='y', length=0, pad=6)
    for spine in family_ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(n_regions))
    ax.set_xticklabels(
        [regional_feature_display_label(feature, tissue) for feature in matrix.index],
        rotation=55,
        ha='right',
        rotation_mode='anchor',
        fontsize=max(9.8, min(11.5, 880.0 / max(n_regions, 1))),
    )
    ax.set_yticks(np.arange(n_metrics))
    ax.tick_params(axis='y', labelleft=False)
    ax.tick_params(length=0, pad=2)
    ax.set_xlabel(
        'Bundle' if tissue == 'wm' else 'Parcel',
        fontsize=14.0,
        fontweight='bold',
    )
    ax.set_ylabel('')
    family_ax.set_ylabel('Metric', fontsize=14.0, fontweight='bold', labelpad=9)
    fig.text(
        0.22,
        0.955,
        f'{REGIONAL_DOMAINS[tissue][1]} {icc_label}',
        ha='left',
        va='top',
        fontsize=21,
        fontweight='bold',
    )

    cbar_ax = fig.add_axes([0.36, 0.07, 0.36, 0.015])
    cbar = fig.colorbar(image, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(icc_label, fontsize=14.0, fontweight='bold', labelpad=6)
    cbar.ax.xaxis.set_label_position('top')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=12.0, length=4)

    observed_sources = [key for key in SOURCE_IMAGE_COLORS if key in set(metric_sources)]
    handles = [
        Patch(
            facecolor=SOURCE_IMAGE_COLORS[key],
            edgecolor='none',
            label=source_image_display_label(key),
        )
        for key in observed_sources
    ]
    legend = fig.legend(
        handles=handles,
        title=METRIC_FAMILY_LEGEND_TITLE,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.025),
        ncol=max(1, len(handles)),
        frameon=False,
        fontsize=12.5,
        title_fontsize=13.5,
        handlelength=1.5,
        columnspacing=1.25,
    )
    legend.get_title().set_fontweight('bold')
    position_regional_guides(fig, ax, cbar_ax, legend)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    save_order_table(output_stem, row_means, column_means, display)
    save_coverage_table(output_stem, matrix, display)
    for extension in ('pdf', 'png'):
        output = output_stem.with_suffix(f'.{extension}')
        fig.savefig(output, dpi=240, bbox_inches='tight')
        print(f'Wrote: {output}')
    plt.close(fig)


def plot_voxel_intervals(
    voxel_data: pd.DataFrame,
    tissue: str,
    output_stem: Path,
) -> None:
    summary = summarize_metric_values(voxel_data)
    summary = summary.loc[summary['tissue'] == tissue].copy()
    summary = summary.sort_values(['median', 'metric'], ascending=[False, True])
    if summary.empty:
        raise RuntimeError(f'No finite voxelwise ICC values for {tissue}.')

    column_tables = [chunk.copy() for chunk in np.array_split(summary, 2) if not chunk.empty]
    max_column_metrics = max(len(chunk) for chunk in column_tables)
    fig_height = max(8.0, 2.15 + 0.255 * max_column_metrics)
    fig, axes = plt.subplots(
        1,
        len(column_tables),
        figsize=(14.5, fig_height),
        constrained_layout=False,
        squeeze=False,
    )
    axes = list(axes[0])
    for ax, column_table in zip(axes, column_tables, strict=True):
        positions = np.arange(len(column_table))
        for position, (_, row) in zip(positions, column_table.iterrows(), strict=True):
            color = SOURCE_IMAGE_COLORS.get(row['source_image'], SOURCE_IMAGE_COLORS['Other'])
            ax.add_patch(
                Rectangle(
                    (row['q25'], position - 0.22),
                    max(row['q75'] - row['q25'], 0.001),
                    0.44,
                    facecolor=color,
                    edgecolor='#2b2b2b',
                    linewidth=0.65,
                    alpha=0.88,
                    zorder=2,
                )
            )
            ax.plot(
                [row['median'], row['median']],
                [position - 0.25, position + 0.25],
                color='white',
                lw=1.8,
                zorder=3,
            )
            ax.scatter(
                [row['median']],
                [position],
                s=22,
                facecolor='white',
                edgecolor='#2b2b2b',
                linewidth=0.65,
                zorder=4,
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
        ax.set_yticklabels(column_table['metric'], fontsize=9.4)
        ax.tick_params(axis='y', length=0, pad=4)
        ax.tick_params(axis='x', labelsize=9.5)
        ax.set_ylim(len(column_table) - 0.2, -0.8)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    xlabel = (
        'ICC(2,1) across white matter voxels'
        if tissue == 'wm'
        else 'ICC(2,1) across cortical gray matter voxels'
    )
    fig.supxlabel(xlabel, fontsize=11.5, fontweight='bold', y=0.052)
    fig.suptitle(
        VOXEL_TITLES[tissue],
        x=0.20,
        y=0.985,
        ha='left',
        fontsize=17,
        fontweight='bold',
    )

    sources = [key for key in SOURCE_IMAGE_COLORS if key in set(summary['source_image'])]
    handles = [
        Patch(
            facecolor=SOURCE_IMAGE_COLORS[key],
            edgecolor='none',
            label=source_image_display_label(key),
        )
        for key in sources
    ]
    fig.subplots_adjust(left=0.20, right=0.985, top=0.955, bottom=0.105, wspace=0.48)
    legend = fig.legend(
        handles=handles,
        title=METRIC_FAMILY_LEGEND_TITLE,
        loc='lower center',
        bbox_to_anchor=(0.59, 0.002),
        ncol=max(1, len(handles)),
        frameon=False,
        fontsize=9.5,
        title_fontsize=10.5,
        handlelength=1.4,
        columnspacing=1.15,
    )
    legend.get_title().set_fontweight('bold')

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_stem.with_suffix('.summary.tsv'), sep='\t', index=False)
    for extension in ('pdf', 'png'):
        output = output_stem.with_suffix(f'.{extension}')
        fig.savefig(output, dpi=240, bbox_inches='tight')
        print(f'Wrote: {output}')
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-set', choices=('primary', 'full'), default='full')
    parser.add_argument(
        '--tissue',
        choices=('wm', 'gm', 'both'),
        default='both',
        help='Tissue/domain to render. The default renders both.',
    )
    parser.add_argument('--stat', choices=('mean', 'median'), default='median')
    parser.add_argument(
        '--icc-column',
        choices=('ICC2_1', 'ICC3_1'),
        default='ICC2_1',
        help='ICC column for parcel/bundle heatmaps; voxelwise maps are ICC(2,1).',
    )
    parser.add_argument('--mni-icc-dir', type=Path, default=default_mni_icc_dir())
    parser.add_argument(
        '--parcel-bundle-icc-dir',
        type=Path,
        default=default_parcel_bundle_icc_dir(),
    )
    parser.add_argument(
        '--patterns-file',
        type=Path,
        default=CODE_ROOT / 'configuration' / 'metrics.yml',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=RUN_FIGURES_ROOT / 'supplemental_icc',
    )
    parser.add_argument(
        '--output-name',
        default=None,
        help='Exact output stem for a single requested ICC plot.',
    )
    parser.add_argument('--voxelwise-analysis', default='primary')
    parser.add_argument('--max-voxels-per-metric', type=int, default=100000)
    parser.add_argument('--skip-regional', action='store_true')
    parser.add_argument('--skip-voxelwise', action='store_true')
    parser.add_argument('--strict', action='store_true')
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Render selected expanded-set ICC panels."""

    args = build_parser().parse_args(argv)
    require_dependencies(need_nibabel=not args.skip_voxelwise)
    mpl.rcParams.update(
        {
            'font.family': 'Arial',
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        }
    )
    patterns_file = args.patterns_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    tissues = ('wm', 'gm') if args.tissue == 'both' else (args.tissue,)

    if not args.skip_regional:
        for tissue in tissues:
            input_path = regional_input_path(
                args.parcel_bundle_icc_dir.expanduser().resolve(),
                tissue,
                args.analysis_set,
                args.stat,
            )
            if not input_path.exists():
                message = f'Missing regional ICC input, skipping: {input_path}'
                if args.strict:
                    raise FileNotFoundError(message)
                print(f'[WARN] {message}', file=sys.stderr)
                continue
            table = load_regional_icc(
                input_path,
                tissue,
                patterns_file,
                args.analysis_set,
                args.icc_column,
            )
            icc_label = icc_display_label(args.icc_column)
            output_stem = output_dir / (
                args.output_name
                or f'icc_{args.analysis_set}_{tissue}_{REGIONAL_DOMAINS[tissue][0]}_heatmap'
            )
            plot_regional_heatmap(table, tissue, args.icc_column, icc_label, output_stem)

    if not args.skip_voxelwise:
        voxel_data = load_voxelwise_icc(
            args.mni_icc_dir.expanduser().resolve(),
            args.analysis_set,
            args.voxelwise_analysis,
            patterns_file,
            args.max_voxels_per_metric,
        )
        for tissue in tissues:
            tissue_data = voxel_data.loc[voxel_data['tissue'] == tissue].copy()
            if tissue_data.empty:
                message = f'No voxelwise ICC data for {tissue}; skipping.'
                if args.strict:
                    raise RuntimeError(message)
                print(f'[WARN] {message}', file=sys.stderr)
                continue
            plot_voxel_intervals(
                tissue_data,
                tissue,
                output_dir / (args.output_name or f'icc_{args.analysis_set}_{tissue}_voxels'),
            )


if __name__ == '__main__':
    main()
