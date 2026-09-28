#!/usr/bin/env python3
"""Plot full-metric WM-bundle and GM-parcel discriminability as faceted heatmaps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    mpl = None
    plt = None
    np = None
    pd = None

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.metrics import (  # noqa: E402
    MetricSpec,
    build_metric_specs,
    metric_plot_label,
    noddi_hybrid_label,
)
from utils.paths import CODE_ROOT, OUTPUT_DERIVATIVES_ROOT, RUN_FIGURES_ROOT  # noqa: E402

from figures._shared.discriminability import (  # noqa: E402
    SCORE_COLUMNS,
    default_gm_input,
    default_wm_input,
    load_discriminability_table,
    require_dependencies,
)


CATEGORY_LABELS = {
    'Tensor': 'Tensor Metrics',
    'DKI': 'DKI',
    'DKI Micro': 'DKI Microstructure',
    'NODDI': 'NODDI',
    'MAPMRI': 'MAP-MRI',
    'GQI': 'DSI Studio GQI',
    'dMRI': 'Other dMRI',
    'T1w/T2w': 'T₁w/T₂w',
    'g-ratio': r'$\it{g}$-Ratio',
    'R1': 'MP2RAGE R₁',
    'MESE': 'R₂',
    'MEGRE': 'MEGRE',
    'Q-Ratio': r'$\it{q}$-Ratio',
    'ihMT': 'ihMT',
    'QSM': 'QSM',
}


def supplemental_family(spec: MetricSpec) -> str:
    """Return the publication-facing family used in this supplemental plot."""

    if spec.group == 'Q-Ratio':
        return 'Q-Ratio'
    if spec.group == 'dMRI' and (
        '(DSIStudio)' in spec.pattern_key
        or 'TORTOISE; Inner Shells' in spec.pattern_key
        or 'TORTOISE; Full Shells' in spec.pattern_key
    ):
        return 'Tensor'
    return spec.family


def compact_metric_label(spec: MetricSpec, category: str) -> str:
    """Remove information already supplied by the facet title."""

    key = spec.pattern_key
    if category == 'DKI Micro':
        return key.removeprefix('DKI Micro ').replace('AxonalD', 'Axonal D')
    if category == 'DKI':
        return key.removeprefix('DKI ')
    if category == 'GQI':
        return key.removeprefix('GQI ')
    if category == 'NODDI':
        return metric_plot_label(noddi_hybrid_label(key))
    if category == 'Tensor':
        for suffix, software in (
            (' (DSIStudio)', 'DSI Studio'),
            (' (TORTOISE; Inner Shells)', 'TORTOISE inner'),
            (' (TORTOISE; Full Shells)', 'TORTOISE full'),
        ):
            if key.endswith(suffix):
                return f'{key.removesuffix(suffix)} ({software})'
    if category == 'T1w/T2w':
        return key.removesuffix('-MyelinW')
    if category == 'g-ratio':
        return key.removeprefix('G-')
    if category == 'Q-Ratio':
        return metric_plot_label(key.removeprefix('Q-Ratio-'))
    if category == 'QSM':
        label = metric_plot_label(spec.label)
        return label.removeprefix('QSM-')
    return metric_plot_label(spec.label)


def facet_metric_key(spec: MetricSpec) -> str:
    """Collapse tissue-specific implementations of one conceptual metric."""

    if spec.family == 'NODDI':
        return noddi_hybrid_label(spec.pattern_key)
    return spec.label


def metric_categories(patterns_file: Path, level: str) -> tuple[dict[str, str], list[str]]:
    specs = build_metric_specs(patterns_file)
    if level == 'family':
        lookup = {spec.label: supplemental_family(spec) for spec in specs}
        order = list(dict.fromkeys(supplemental_family(spec) for spec in specs))
    elif level == 'group':
        lookup = {spec.label: spec.group for spec in specs}
        order = list(dict.fromkeys(spec.group for spec in specs))
    elif level == 'source-image':
        lookup = {spec.label: spec.source_image for spec in specs}
        order = list(dict.fromkeys(spec.source_image for spec in specs))
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(level)
    return lookup, order


def add_categories(
    data: pd.DataFrame,
    patterns_file: Path,
    level: str,
) -> tuple[pd.DataFrame, list[str]]:
    specs = build_metric_specs(patterns_file)
    spec_by_label = {spec.label: spec for spec in specs}
    lookup, registry_order = metric_categories(patterns_file, level)
    out = data.copy()
    out['category'] = out['metric_key'].map(lookup).fillna('Other')
    out['facet_metric_key'] = out['metric_key'].map(
        lambda label: facet_metric_key(spec_by_label[label]) if label in spec_by_label else label
    )
    out['facet_metric_label'] = out.apply(
        lambda row: (
            compact_metric_label(spec_by_label[row['metric_key']], row['category'])
            if row['metric_key'] in spec_by_label
            else str(row['metric'])
        ),
        axis=1,
    )
    observed = set(out['category'])
    order = [category for category in registry_order if category in observed]
    if 'Other' in observed:
        order.append('Other')
    return out, order


def inclusion_path(score_path: Path) -> Path:
    return score_path.with_name(f'{score_path.stem}_metric_inclusion.tsv')


def report_unexpected_missing_scores(
    data: pd.DataFrame,
    patterns_file: Path,
    score_paths: dict[str, Path],
) -> None:
    """Explain tissue gaps using the metric-inclusion files when available."""

    specs = build_metric_specs(patterns_file)
    expected: dict[str, set[str]] = {}
    raw_keys: dict[tuple[str, str], list[str]] = {}
    for spec in specs:
        concept = facet_metric_key(spec)
        expected.setdefault(concept, set()).update(
            tissue for tissue in ('wm', 'gm') if tissue in spec.tissues
        )
        for tissue in ('wm', 'gm'):
            if tissue in spec.tissues:
                raw_keys.setdefault((concept, tissue), []).append(spec.label)

    observed = (
        data.groupby('facet_metric_key', observed=True)['tissue']
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )
    inclusion_tables: dict[str, pd.DataFrame] = {}
    for tissue, score_path in score_paths.items():
        path = inclusion_path(score_path)
        if path.exists():
            inclusion_tables[tissue] = pd.read_csv(path, sep='\t')

    for concept, expected_tissues in expected.items():
        # Report asymmetric gaps visible in the figure, not metrics omitted
        # altogether by a smaller analysis set.
        if concept not in observed:
            continue
        missing = expected_tissues - observed.get(concept, set())
        for tissue in sorted(missing):
            details = ''
            inclusion = inclusion_tables.get(tissue)
            if inclusion is not None:
                candidates = raw_keys.get((concept, tissue), [])
                rows = inclusion.loc[inclusion['metric_key'].isin(candidates)]
                if not rows.empty:
                    reasons = sorted(
                        {
                            str(reason)
                            for reason in rows['reason_if_not_scored'].dropna()
                            if str(reason)
                        }
                    )
                    if reasons:
                        details = f' ({"; ".join(reasons)})'
            print(
                f'[WARN] No {tissue.upper()} discriminability score for {concept}{details}',
                flush=True,
            )


def pack_categories(
    categories: list[str],
    counts: dict[str, int],
    max_columns_per_row: int,
) -> list[list[str]]:
    rows: list[list[str]] = []
    current: list[str] = []
    current_columns = 0
    for category in categories:
        # Reserve a little horizontal space for each panel's title and row
        # labels so a run of one- or two-metric families does not collide.
        n_columns = max(1, counts[category]) + 2
        if current and current_columns + n_columns > max_columns_per_row:
            rows.append(current)
            current = []
            current_columns = 0
        current.append(category)
        current_columns += n_columns
    if current:
        rows.append(current)
    return rows


def category_matrix(
    data: pd.DataFrame,
    category: str,
) -> tuple[pd.DataFrame, dict[str, str], pd.Series]:
    category_data = data.loc[data['category'] == category].copy()
    matrix = category_data.pivot_table(
        index='tissue',
        columns='facet_metric_key',
        values='score',
        aggfunc='first',
    ).reindex(index=['wm', 'gm'])
    means = matrix.mean(axis=0, skipna=True).sort_values(ascending=False)
    matrix = matrix.reindex(columns=means.index)
    display = (
        category_data.drop_duplicates('facet_metric_key')
        .set_index('facet_metric_key')['facet_metric_label']
        .astype(str)
        .to_dict()
    )
    return matrix, display, means


def plot_faceted_heatmaps(
    data: pd.DataFrame,
    categories: list[str],
    score_label: str,
    output_stem: Path,
    max_columns_per_row: int,
) -> None:
    if data.empty:
        raise RuntimeError('No finite discriminability values to plot.')
    finite_scores = data['score'].to_numpy(dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if finite_scores.size == 0:
        raise RuntimeError('No finite discriminability values to plot.')
    observed_min = float(np.min(finite_scores))
    color_min = max(0.0, min(0.90, np.floor(observed_min * 20.0) / 20.0))
    color_ticks = np.linspace(color_min, 1.0, 6)
    print(
        f'[INFO] Discriminability color scale: {color_min:.2f} to 1.00 '
        f'(observed minimum {observed_min:.3f}).',
        flush=True,
    )
    counts = {
        category: int(data.loc[data['category'] == category, 'facet_metric_key'].nunique())
        for category in categories
    }
    category_rows = pack_categories(categories, counts, max_columns_per_row)
    fig = plt.figure(
        figsize=(17.0, max(7.0, 4.55 * len(category_rows) + 1.6)),
        constrained_layout=False,
    )
    outer = fig.add_gridspec(
        len(category_rows) + 1,
        1,
        height_ratios=[1.0] * len(category_rows) + [0.075],
        left=0.065,
        right=0.985,
        top=0.94,
        bottom=0.06,
        hspace=1.50,
    )
    cmap = mpl.colormaps['viridis'].copy()
    cmap.set_bad('#e5e5e5')
    image = None
    ordering_records: list[dict[str, object]] = []

    for row_index, row_categories in enumerate(category_rows):
        inner = outer[row_index].subgridspec(
            1,
            len(row_categories),
            width_ratios=[max(2.5, counts[category]) for category in row_categories],
            wspace=0.28,
        )
        for panel_index, category in enumerate(row_categories):
            ax = fig.add_subplot(inner[0, panel_index])
            matrix, display, means = category_matrix(data, category)
            image = ax.imshow(
                matrix.to_numpy(dtype=float),
                aspect='auto',
                interpolation='nearest',
                cmap=cmap,
                vmin=color_min,
                vmax=1.0,
            )
            for y_index in range(matrix.shape[0]):
                for x_index in range(matrix.shape[1]):
                    value = matrix.iloc[y_index, x_index]
                    if np.isfinite(value):
                        normalized = np.clip(
                            (float(value) - color_min) / max(1.0 - color_min, 1e-12),
                            0.0,
                            1.0,
                        )
                        red, green, blue, _ = cmap(normalized)
                        luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                        ax.text(
                            x_index,
                            y_index,
                            f'{value:.2f}',
                            ha='center',
                            va='center',
                            fontsize=13.0,
                            color='white' if luminance < 0.48 else '#111111',
                            fontweight='bold',
                        )
            ax.set_xticks(np.arange(matrix.shape[1]))
            ax.set_xticklabels(
                [display.get(metric, metric) for metric in matrix.columns],
                rotation=52,
                ha='right',
                rotation_mode='anchor',
                fontsize=max(10.5, min(13.0, 120.0 / max(matrix.shape[1], 1))),
            )
            ax.set_yticks([0, 1])
            ax.set_yticklabels(
                ['White matter bundles', 'Gray matter parcels'] if panel_index == 0 else ['', ''],
                fontsize=12.0,
            )
            ax.tick_params(length=0, pad=4)
            ax.set_title(
                CATEGORY_LABELS.get(category, category.replace('_', ' ').title()),
                loc='left',
                fontsize=16.0,
                fontweight='bold',
                pad=8,
            )
            for spine in ax.spines.values():
                spine.set_edgecolor('#555555')
                spine.set_linewidth(0.6)

            for rank, (metric, mean_score) in enumerate(means.items(), start=1):
                ordering_records.append(
                    {
                        'category': category,
                        'rank_within_category': rank,
                        'facet_metric_key': metric,
                        'metric': display.get(metric, metric),
                        'mean_wm_gm_score': mean_score,
                    }
                )

    if image is None:
        raise RuntimeError('No discriminability panels were drawn.')
    cbar_ax = fig.add_subplot(outer[-1, 0])
    cbar = fig.colorbar(image, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks(color_ticks)
    cbar.ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    cbar.ax.tick_params(labelsize=12.0, length=4)
    cbar.set_label(score_label, fontsize=14.0, fontweight='bold', labelpad=7)
    fig.suptitle(
        f'White Matter Bundle and Gray Matter Parcel {score_label}',
        fontsize=22,
        fontweight='bold',
        y=0.975,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_stem.with_suffix('.summary.tsv'), sep='\t', index=False)
    pd.DataFrame(ordering_records).to_csv(
        output_stem.with_name(output_stem.name + '_ordering.tsv'),
        sep='\t',
        index=False,
    )
    for extension in ('pdf', 'png'):
        output = output_stem.with_suffix(f'.{extension}')
        fig.savefig(output, dpi=260, bbox_inches='tight')
        print(f'Wrote: {output}')
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-set', choices=('primary', 'full'), default='full')
    parser.add_argument('--stat', choices=('mean', 'median'), default='median')
    parser.add_argument(
        '--distance-metric', choices=('correlation', 'euclidean'), default='correlation'
    )
    parser.add_argument('--score-column', choices=tuple(SCORE_COLUMNS), default='discriminability')
    parser.add_argument(
        '--category-level',
        choices=('family', 'group', 'source-image'),
        default='family',
        help='Registry field used to create separate metric-category heatmaps.',
    )
    parser.add_argument('--max-columns-per-row', type=int, default=28)
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=OUTPUT_DERIVATIVES_ROOT / 'parcel_bundle_discriminability',
    )
    parser.add_argument('--wm-input', type=Path, default=None)
    parser.add_argument('--gm-input', type=Path, default=None)
    parser.add_argument(
        '--patterns-file',
        type=Path,
        default=CODE_ROOT / 'configuration' / 'metrics.yml',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=RUN_FIGURES_ROOT / 'figure_s12_discriminability',
    )
    parser.add_argument('--output-name', default='FigureS12')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_dependencies()
    mpl.rcParams.update(
        {
            'font.family': 'Arial',
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        }
    )
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
    patterns_file = args.patterns_file.expanduser().resolve()
    data, categories = add_categories(
        data,
        patterns_file,
        args.category_level,
    )
    report_unexpected_missing_scores(
        data,
        patterns_file,
        {'wm': wm_input, 'gm': gm_input},
    )
    output_name = args.output_name
    plot_faceted_heatmaps(
        data,
        categories,
        SCORE_COLUMNS[args.score_column],
        args.output_dir.expanduser().resolve() / output_name,
        args.max_columns_per_row,
    )


if __name__ == '__main__':
    main()
