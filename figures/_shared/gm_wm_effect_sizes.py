#!/usr/bin/env python3
"""Plot MNI voxelwise cortical-GM-vs-WM effect sizes by metric."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

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

from utils.metrics import (
    METRIC_FAMILY_LEGEND_TITLE,
    SOURCE_IMAGE_COLORS,
    build_metric_specs,
    metric_plot_label,
    source_image_display_label,
)
from utils.paths import CODE_ROOT, OUTPUT_DERIVATIVES_ROOT, RUN_FIGURES_ROOT

from figures.figure_s12_discriminability.plot_figure_s12_discriminability import (
    CATEGORY_LABELS,
    compact_metric_label,
    pack_categories,
    supplemental_family,
)


EFFECT_LABELS = {
    'robust_median_d': r'Average White Matter–Gray Matter separation (robust $d$)',
    'cohen_d': "Average White Matter–Gray Matter separation (Cohen's d)",
    'hedges_g': "Average White Matter–Gray Matter separation (Hedges' g)",
    'signed_auc': 'Average White Matter–Gray Matter separation (signed AUC)',
    'median_difference': 'Median White Matter − Gray Matter difference',
    'mean_difference': 'Mean White Matter − Gray Matter difference',
    'percent_median_difference': (
        'Median White Matter − Gray Matter difference (% of |White Matter median|)'
    ),
}
GM_TISSUE_LABELS = {
    'cortical_gm': 'Cortical Gray Matter',
    'deep_gm': 'Deep Gray Matter',
    'all_gm': 'All Gray Matter',
}
PRIMARY_GM_TISSUE_LABELS = {
    'cortical_gm': 'Cortical GM',
    'deep_gm': 'Deep GM',
    'all_gm': 'All GM',
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


def load_subject_effects(path: Path, effect: str, gm_tissue: str) -> pd.DataFrame:
    data = pd.read_csv(path, sep='\t')
    required = {
        'gm_tissue',
        'metric_key',
        'display_metric',
        'source_image',
        'subject',
        effect,
    }
    missing = required - set(data.columns)
    if missing:
        raise RuntimeError(f'{path} is missing required columns: {", ".join(sorted(missing))}')
    data[effect] = pd.to_numeric(data[effect], errors='coerce')
    data = data.loc[data['gm_tissue'].astype(str) == gm_tissue].copy()
    data = data.loc[data['source_image'].astype(str) != 'g-ratio'].copy()
    if data.empty:
        raise RuntimeError(f'No {gm_tissue} rows found in {path}')
    return data.dropna(subset=[effect]).copy()


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 10000) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample = rng.choice(finite, size=finite.size, replace=True)
        estimates[index] = np.mean(sample)
    return tuple(float(value) for value in np.percentile(estimates, [2.5, 97.5]))


def summarize_for_plot(data: pd.DataFrame, effect: str) -> pd.DataFrame:
    rows = []
    for group_index, ((metric_key, display_metric, source_image), group) in enumerate(
        data.groupby(
            ['metric_key', 'display_metric', 'source_image'],
            sort=False,
        )
    ):
        values = -group[effect].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        ci_low, ci_high = bootstrap_ci(values, seed=20260819 + group_index)
        rows.append(
            {
                'metric_key': metric_key,
                'display_metric': display_metric,
                'source_image': source_image,
                'n_subjects': int(group['subject'].nunique()),
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
                'ci95_low': ci_low,
                'ci95_high': ci_high,
            }
        )
    summary = pd.DataFrame(rows)
    summary['abs_mean'] = summary['mean'].abs()
    summary = summary.sort_values(['abs_mean', 'display_metric']).drop(columns=['abs_mean'])
    return summary.reset_index(drop=True)


def axis_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    low, high = np.percentile(finite, [1, 99])
    low = min(low, float(np.min(finite)))
    high = max(high, float(np.max(finite)))
    span = max(high - low, 1e-6)
    pad = 0.08 * span
    low -= pad
    high += pad
    if low < 0 < high:
        max_abs = max(abs(low), abs(high))
        return -max_abs, max_abs
    return low, high


def display_effect_values(values: np.ndarray) -> np.ndarray:
    return -np.asarray(values, dtype=float)


def display_metric_label(label: str) -> str:
    label = 'ICVF†' if label == 'ICVF' else label
    return metric_plot_label(label)


def add_effect_categories(
    summary: pd.DataFrame,
    patterns_file: Path,
) -> tuple[pd.DataFrame, list[str]]:
    specs = build_metric_specs(patterns_file)
    spec_by_label = {spec.label: spec for spec in specs}
    out = summary.copy()
    out['category'] = out['metric_key'].map(
        lambda key: supplemental_family(spec_by_label[key]) if key in spec_by_label else 'Other'
    )
    out['compact_metric'] = out.apply(
        lambda row: (
            compact_metric_label(spec_by_label[row['metric_key']], row['category'])
            if row['metric_key'] in spec_by_label
            else display_metric_label(str(row['display_metric']))
        ),
        axis=1,
    )
    registry_order = list(
        dict.fromkeys(supplemental_family(spec) for spec in specs if spec.source_image != 'g-ratio')
    )
    observed = set(out['category'])
    categories = [category for category in registry_order if category in observed]
    if 'Other' in observed:
        categories.append('Other')
    return out, categories


def plot_faceted_effect_sizes(
    data: pd.DataFrame,
    out_prefix: Path,
    effect: str,
    gm_tissue: str,
    patterns_file: Path,
    max_columns_per_row: int,
) -> None:
    summary = summarize_for_plot(data, effect)
    if summary.empty:
        raise RuntimeError(f'No finite {effect} values to plot.')
    summary, categories = add_effect_categories(summary, patterns_file)
    counts = {
        category: int(summary.loc[summary['category'] == category, 'metric_key'].nunique())
        for category in categories
    }
    category_rows = pack_categories(categories, counts, max_columns_per_row)

    finite = summary['mean'].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    max_abs = max(0.25, float(np.max(np.abs(finite))))
    color_limit = float(np.ceil(max_abs * 2.0) / 2.0)
    cmap = mpl.colormaps['RdBu_r'].copy()

    fig = plt.figure(
        figsize=(17.0, max(7.0, 3.65 * len(category_rows) + 1.7)),
        constrained_layout=False,
    )
    outer = fig.add_gridspec(
        len(category_rows) + 1,
        1,
        height_ratios=[1.0] * len(category_rows) + [0.09],
        left=0.055,
        right=0.985,
        top=0.925,
        bottom=0.065,
        hspace=1.35,
    )
    image = None
    ordering_records: list[dict[str, object]] = []

    for row_index, row_categories in enumerate(category_rows):
        inner = outer[row_index].subgridspec(
            1,
            len(row_categories),
            width_ratios=[max(2.5, counts[category]) for category in row_categories],
            wspace=0.30,
        )
        for panel_index, category in enumerate(row_categories):
            ax = fig.add_subplot(inner[0, panel_index])
            category_data = summary.loc[summary['category'] == category].copy()
            category_data['abs_mean'] = category_data['mean'].abs()
            category_data = category_data.sort_values(
                ['abs_mean', 'display_metric'],
                ascending=[False, True],
            ).reset_index(drop=True)
            values = category_data['mean'].to_numpy(dtype=float).reshape(1, -1)
            image = ax.imshow(
                values,
                aspect='auto',
                interpolation='nearest',
                cmap=cmap,
                vmin=-color_limit,
                vmax=color_limit,
            )
            for x_index, value in enumerate(values[0]):
                annotation_color = 'white' if abs(value) >= 0.58 * color_limit else '#111111'
                ax.text(
                    x_index,
                    0,
                    f'{value:.2f}',
                    ha='center',
                    va='center',
                    fontsize=13.0,
                    color=annotation_color,
                    fontweight='bold',
                )
            ax.set_xticks(np.arange(len(category_data)))
            ax.set_xticklabels(
                category_data['compact_metric'].tolist(),
                rotation=52,
                ha='right',
                rotation_mode='anchor',
                fontsize=max(10.5, min(13.0, 120.0 / max(len(category_data), 1))),
            )
            ax.set_yticks([])
            ax.tick_params(length=0, pad=4)
            category_title = CATEGORY_LABELS.get(category, category.replace('_', ' ').title())
            if category == 'NODDI':
                category_title = r'NODDI$^{\dagger}$'
            ax.set_title(
                category_title,
                loc='left',
                fontsize=16.0,
                fontweight='bold',
                pad=8,
            )
            for spine in ax.spines.values():
                spine.set_edgecolor('#555555')
                spine.set_linewidth(0.6)

            for rank, row in category_data.iterrows():
                ordering_records.append(
                    {
                        'category': category,
                        'rank_by_absolute_effect': rank + 1,
                        'metric_key': row['metric_key'],
                        'metric': row['display_metric'],
                        'mean_effect': row['mean'],
                        'absolute_mean_effect': row['abs_mean'],
                    }
                )

    if image is None:
        raise RuntimeError('No effect-size panels were drawn.')
    cbar_ax = fig.add_subplot(outer[-1, 0])
    cbar = fig.colorbar(image, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks(np.linspace(-color_limit, color_limit, 5))
    gm_label = GM_TISSUE_LABELS[gm_tissue]
    effect_label = EFFECT_LABELS.get(effect, effect).replace('Gray Matter', gm_label)
    cbar.ax.tick_params(labelsize=12.0, length=4)
    cbar.set_label(effect_label, fontsize=14.0, fontweight='bold', labelpad=7)
    cbar.ax.text(
        0.0,
        1.95,
        f'{gm_label} > White Matter',
        transform=cbar.ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=13.0,
    )
    cbar.ax.text(
        1.0,
        1.95,
        f'White Matter > {gm_label}',
        transform=cbar.ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=13.0,
    )
    fig.suptitle(
        'Gray–White Matter Differentiation',
        fontsize=22,
        fontweight='bold',
        y=0.97,
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_prefix.with_suffix('.summary.tsv'), sep='\t', index=False)
    pd.DataFrame(ordering_records).to_csv(
        out_prefix.with_name(out_prefix.name + '_ordering.tsv'),
        sep='\t',
        index=False,
    )
    for extension in ('png', 'pdf'):
        out_file = out_prefix.with_suffix(f'.{extension}')
        fig.savefig(out_file, bbox_inches='tight', dpi=300)
        print(f'Wrote: {out_file}')
    plt.close(fig)


def plot_effect_sizes(
    data: pd.DataFrame,
    out_prefix: Path,
    effect: str,
    gm_tissue: str,
    show_subject_points: bool,
) -> None:
    summary = summarize_for_plot(data, effect)
    if summary.empty:
        raise RuntimeError(f'No finite {effect} values to plot.')

    order = summary['metric_key'].tolist()
    y_step = 0.72
    y = np.arange(len(order)) * y_step
    order_lookup = dict(zip(order, y, strict=True))
    fig_height = max(5.9, 0.24 * len(order) + 1.9)
    fig, ax = plt.subplots(figsize=(6.9, fig_height), constrained_layout=False)

    rng = np.random.default_rng(20260818)
    for _, row in summary.iterrows():
        y_pos = order_lookup[row['metric_key']]
        color = SOURCE_IMAGE_COLORS.get(row['source_image'], SOURCE_IMAGE_COLORS['Other'])
        ax.barh(
            y_pos,
            row['mean'],
            height=0.42,
            left=0,
            color=color,
            edgecolor='none',
            alpha=0.86,
            zorder=2,
        )
        if np.isfinite(row['ci95_low']) and np.isfinite(row['ci95_high']):
            ax.hlines(
                y_pos, row['ci95_low'], row['ci95_high'], color='#1f1f1f', linewidth=1.0, zorder=3
            )
            ax.plot(
                [row['ci95_low'], row['ci95_low']],
                [y_pos - 0.105, y_pos + 0.105],
                color='#1f1f1f',
                lw=0.8,
                zorder=3,
            )
            ax.plot(
                [row['ci95_high'], row['ci95_high']],
                [y_pos - 0.105, y_pos + 0.105],
                color='#1f1f1f',
                lw=0.8,
                zorder=3,
            )
        ax.scatter(
            [row['mean']],
            [y_pos],
            s=38,
            facecolor='white',
            edgecolor='#1f1f1f',
            linewidth=0.8,
            zorder=5,
        )
        if show_subject_points:
            metric_values = display_effect_values(
                data.loc[data['metric_key'] == row['metric_key'], effect].to_numpy(dtype=float)
            )
            jitter = rng.uniform(-0.10, 0.10, size=metric_values.size)
            ax.scatter(
                metric_values,
                y_pos + jitter,
                s=8,
                color='black',
                alpha=0.22,
                linewidth=0,
                zorder=1,
            )
    labels = [display_metric_label(label) for label in summary['display_metric']]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.8)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', labelsize=11.5)
    plot_values = display_effect_values(data[effect].to_numpy(dtype=float))
    if effect in {'robust_median_d', 'cohen_d', 'hedges_g'}:
        plotted_extents = np.concatenate(
            [
                summary['mean'].to_numpy(dtype=float),
                summary['ci95_low'].to_numpy(dtype=float),
                summary['ci95_high'].to_numpy(dtype=float),
            ]
        )
        finite_extents = plotted_extents[np.isfinite(plotted_extents)]
        x_low, x_high = -2.0, 4.0
        if finite_extents.size:
            x_low = min(x_low, float(np.min(finite_extents)) - 0.08)
            x_high = max(x_high, float(np.max(finite_extents)) + 0.08)
    else:
        x_low, x_high = axis_limits(plot_values)
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(float(y[0] - 0.55), float(y[-1] + 0.55))
    for value, color, linewidth in (
        (-3, '#2F5F9E', 1.05),
        (-1, '#A8C8EA', 0.95),
        (1, '#F0A6A6', 0.95),
        (3, '#B12A2A', 1.05),
    ):
        if x_low <= value <= x_high:
            ax.axvline(value, color=color, lw=linewidth, zorder=0)
    ax.axvline(0, color='#6a6a6a', lw=1.0, ls=':', zorder=1)
    ax.grid(False)
    ax.grid(axis='y', visible=False)
    gm_label = PRIMARY_GM_TISSUE_LABELS[gm_tissue]
    effect_label = EFFECT_LABELS.get(effect, effect)
    effect_label = effect_label.replace('Gray Matter', gm_label).replace('White Matter', 'WM')
    ax.set_xlabel(effect_label, fontsize=13.0, labelpad=9)
    ax.set_ylabel('')
    directional_gm_label = 'GM' if gm_tissue == 'cortical_gm' else gm_label
    ax.text(
        0.01,
        1.01,
        f'{directional_gm_label} > WM',
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=11.5,
        color='#333333',
    )
    ax.text(
        0.99,
        1.01,
        f'WM > {directional_gm_label}',
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=11.5,
        color='#333333',
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sources = [
        source
        for source in SOURCE_IMAGE_COLORS
        if source != 'g-ratio' and source in set(summary['source_image'])
    ]
    handles = [
        Patch(
            facecolor=SOURCE_IMAGE_COLORS[source],
            edgecolor='none',
            label=source_display_label(source),
        )
        for source in sources
    ]
    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=min(4, len(handles)),
        title=METRIC_FAMILY_LEGEND_TITLE,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
        fontsize=10.6,
        title_fontsize=11.0,
    )
    bottom_margin = max(0.08, min(0.19, 1.65 / fig_height))
    fig.subplots_adjust(left=0.34, right=0.985, top=0.965, bottom=bottom_margin)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        out_file = out_prefix.with_suffix(f'.{extension}')
        fig.savefig(out_file, bbox_inches='tight', dpi=300)
        print(f'Wrote: {out_file}')
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--analysis-set',
        choices=('primary', 'full'),
        default='primary',
        help='Metric set used to choose default input and output paths.',
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=None,
        help='Subject-averaged effect-size TSV from compute_mni_gm_wm_effect_sizes.py.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output stem. Defaults to a name based on analysis set, GM tissue, and effect.',
    )
    parser.add_argument(
        '--effect',
        choices=tuple(EFFECT_LABELS),
        default='robust_median_d',
    )
    parser.add_argument(
        '--gm-tissue',
        choices=tuple(GM_TISSUE_LABELS),
        default='cortical_gm',
        help='GM compartment to plot. The primary figure uses cortical_gm.',
    )
    parser.add_argument(
        '--show-subject-points',
        action='store_true',
        help='Overlay subject-level points behind the metric summaries.',
    )
    parser.add_argument(
        '--patterns-file',
        type=Path,
        default=CODE_ROOT / 'configuration' / 'metrics.yml',
        help='Metric registry used to organize the full supplemental heatmap.',
    )
    parser.add_argument(
        '--max-columns-per-row',
        type=int,
        default=28,
        help='Approximate maximum number of metric cells per heatmap row.',
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Render the primary or expanded GM/WM effect-size figure."""

    args = parse_args(argv)
    require_dependencies()
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    input_path = args.input or (
        OUTPUT_DERIVATIVES_ROOT
        / 'mni_gm_wm_effect_sizes'
        / f'mni_gm_wm_effect_sizes_{args.analysis_set}_subject.tsv'
    )
    output_stem = args.output or (
        RUN_FIGURES_ROOT
        / 'gm_wm_effect_sizes'
        / f'gm_wm_effect_sizes_{args.analysis_set}_{args.gm_tissue}_{args.effect}'
    )
    data = load_subject_effects(
        input_path.expanduser().resolve(),
        args.effect,
        args.gm_tissue,
    )
    if args.analysis_set == 'full':
        plot_faceted_effect_sizes(
            data,
            output_stem.expanduser().resolve(),
            args.effect,
            args.gm_tissue,
            args.patterns_file.expanduser().resolve(),
            args.max_columns_per_row,
        )
    else:
        plot_effect_sizes(
            data,
            output_stem.expanduser().resolve(),
            args.effect,
            args.gm_tissue,
            args.show_subject_points,
        )


if __name__ == '__main__':
    main()
