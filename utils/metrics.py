"""Shared scalar metric definitions for MIRROR analyses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_METRICS_FILE = Path(__file__).resolve().parents[1] / 'configuration' / 'metrics.yml'
ANALYSIS_SET_CHOICES = ('primary', 'full')


def _load_document(path: Path) -> dict:
    """Load JSON or YAML without requiring PyYAML for the JSON-formatted registry."""

    text = path.read_text(encoding='utf-8')
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(f'PyYAML is required to read {path}.') from exc
        document = yaml.safe_load(text)
    if not isinstance(document, dict):
        raise TypeError(f'Metric configuration must be a mapping: {path}')
    return document


_DEFAULT_DOCUMENT = _load_document(DEFAULT_METRICS_FILE)
_DEFAULT_RECORDS = tuple(_DEFAULT_DOCUMENT['metrics'])


@dataclass(frozen=True)
class MetricSpec:
    label: str
    primary_label: str
    pattern_key: str
    group: str
    family: str
    source_image: str
    qc_modalities: tuple[str, ...]
    tissues: tuple[str, ...]
    primary: bool = False
    primary_order: int | None = None
    expanded_order: int | None = None


PRIMARY_METRIC_LABELS = tuple(
    record['primary_label']
    for record in sorted(
        (record for record in _DEFAULT_RECORDS if record['primary']),
        key=lambda record: record['primary_order'],
    )
)

PRIMARY_PATTERN_KEYS = {
    record['primary_label']: record['pattern_key']
    for record in _DEFAULT_RECORDS
    if record['primary']
}

SOURCE_IMAGE_COLORS = dict(_DEFAULT_DOCUMENT['family_colors'])

METRIC_FAMILY_LEGEND_TITLE = 'Metric family'

SOURCE_IMAGE_DISPLAY_LABELS = {
    'T1w/T2w': 'T₁w/T₂w',
    'g-ratio': r'$\it{g}$-Ratio',
    'R1': 'MP2RAGE',
    'MESE': 'R₂',
}


def source_image_display_label(source: str) -> str:
    """Return the publication-facing label for a source-image/metric-family key."""

    return SOURCE_IMAGE_DISPLAY_LABELS.get(source, source)


def norm_token(text: object) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(text).lower())


def display_label(pattern_key: str) -> str:
    if pattern_key.startswith("QSM-X-R2'-"):
        return pattern_key.replace("R2'", 'R2p')
    if pattern_key == 'NG Perpendicular':
        return 'NG (Perpendicular)'
    if pattern_key == 'NG Parallel':
        return 'NG (Parallel)'
    if pattern_key == 'GQI GFA':
        return 'GFA'
    if pattern_key == 'DKI MKT':
        return 'MKT'
    if pattern_key == 'DKI RK':
        return 'RK'
    if pattern_key == 'DKI Micro AWF':
        return 'AWF'
    return pattern_key


def primary_label(pattern_key: str) -> str:
    if pattern_key == 'ICVF (GM)':
        return 'ICVF'
    for label, key in PRIMARY_PATTERN_KEYS.items():
        if key == pattern_key:
            return label
    return display_label(pattern_key)


def metric_plot_label(label: str) -> str:
    """Return the publication-facing label for figures."""

    replacements = {
        'NG Parallel': 'NG ∥',
        'NG (Parallel)': 'NG ∥',
        'NG (Perpendicular)': 'NG⊥',
        'PAth': 'PAθ',
        'MPRAGE-MyelinW': 'MPRAGE T₁w/T₂w Ratio',
        'SPACE-MyelinW': 'SPACE T₁w/T₂w Ratio',
        'QSM-SEPIA-E5-X': 'QSM-SEPIA-E5-χ',
        'QSM-X-R2p-E5-X': 'QSM-χ-R₂p-E5-χ',
        'QSM-X-R2p-E5-Para': 'QSM-χ-R₂p-E5-Para',
        'QSM-X-R2p-E5-Dia': 'QSM-χ-R₂p-E5-Dia',
    }
    label = replacements.get(label, label)
    # Keep internal identifiers unchanged while using the published software
    # name in presentation-facing labels.
    label = label.replace('DSIStudio', 'DSI Studio')
    if label.startswith('QSM-X-'):
        label = label.replace('QSM-X-', 'QSM-χ-', 1)
    if label.startswith('QSM-') and label.endswith('-X'):
        label = f'{label[:-2]}-χ'
    label = re.sub(
        r'-(para|dia)$',
        lambda match: f'-{match.group(1).capitalize()}',
        label,
        flags=re.IGNORECASE,
    )
    if re.match(r'^q-ratio', label, flags=re.IGNORECASE):
        label = re.sub(
            r'^q-ratio',
            lambda _: r'$\it{q}$-Ratio',
            label,
            count=1,
            flags=re.IGNORECASE,
        )
    elif re.match(r'^g-ratio', label, flags=re.IGNORECASE):
        label = re.sub(
            r'^g-ratio',
            lambda _: r'$\it{g}$-Ratio',
            label,
            count=1,
            flags=re.IGNORECASE,
        )
    elif label in {'G-ihMTsat', 'G-ihMTR'}:
        label = rf'$\it{{g}}$-{label.removeprefix("G-")}'
    for source, target in (
        ('R2p', 'R₂p'),
        ('R2*', 'R₂*'),
        ('R2', 'R₂'),
        ('R1', 'R₁'),
        ('B1c', 'B₁c'),
        ('T1w', 'T₁w'),
        ('T2w', 'T₂w'),
    ):
        label = label.replace(source, target)
    return label


def infer_family(group: str, pattern_key: str) -> str:
    if group == 'dMRI':
        if 'TORTOISE; Inner Shells' in pattern_key:
            return 'Tensor'
        if pattern_key.startswith('DKI Micro'):
            return 'DKI Micro'
        if pattern_key.startswith('DKI '):
            return 'DKI'
        if (
            pattern_key.startswith('ICVF')
            or pattern_key.startswith('ISOVF')
            or pattern_key.startswith('OD')
        ):
            return 'NODDI'
        if pattern_key in {
            'NG',
            'NG Parallel',
            'NG Perpendicular',
            'PA',
            'PAth',
            'RTAP',
            'RTOP',
            'RTPP',
        }:
            return 'MAPMRI'
        if pattern_key.startswith('GQI '):
            return 'GQI'
        return 'dMRI'
    if group == 'T1w/T2w Ratio':
        return 'T1w/T2w'
    if group == 'G-Ratio':
        return 'g-ratio'
    if group == 'MP2RAGE':
        return 'R1'
    if group in {'MESE', 'MEGRE'}:
        return group
    if group == 'Q-Ratio':
        return 'MEGRE'
    return group


def source_image_from_group(group: str) -> str:
    if group == 'dMRI':
        return 'dMRI'
    if group == 'T1w/T2w Ratio':
        return 'T1w/T2w'
    if group == 'G-Ratio':
        return 'g-ratio'
    if group == 'MP2RAGE':
        return 'R1'
    if group in {'MESE', 'MEGRE'}:
        return group
    if group == 'Q-Ratio':
        return 'MEGRE'
    if group == 'ihMT':
        return 'ihMT'
    if group == 'QSM':
        return 'QSM'
    return 'Other'


def load_patterns(path: Path) -> dict[str, dict[str, str]]:
    document = _load_document(path)
    if 'metrics' not in document:
        return document
    nested: dict[str, dict[str, str]] = {}
    for record in document['metrics']:
        nested.setdefault(record['group'], {})[record['pattern_key']] = record['pattern']
    return nested


def flatten_metric_patterns(path: Path = DEFAULT_METRICS_FILE) -> dict[str, str]:
    """Return ``pattern_key -> relative glob`` from either registry format."""

    return {
        key: value
        for group_patterns in load_patterns(path).values()
        for key, value in group_patterns.items()
    }


def qc_modalities_for(group: str, pattern_key: str) -> tuple[str, ...]:
    if group == 'dMRI':
        return ('dMRI',)
    if group == 'ihMT':
        if pattern_key in {'ihMTsat', 'ihMTsat-B1c'}:
            return ('MP2RAGE', 'ihMTRAGE', 'B1+')
        return ('ihMTRAGE',)
    if group == 'MP2RAGE':
        if pattern_key == 'R1-B1c':
            return ('MP2RAGE', 'B1+')
        return ('MP2RAGE',)
    if group == 'T1w/T2w Ratio':
        if pattern_key in {'MPRAGE-MyelinW', 'Scaled MPRAGE-MyelinW'}:
            return ('MPRAGE T1w', 'SPACE T2w')
        if pattern_key in {'SPACE-MyelinW', 'Scaled SPACE-MyelinW'}:
            return ('SPACE T1w', 'SPACE T2w')
    if group == 'G-Ratio':
        if pattern_key == 'G-ihMTsat':
            return ('MP2RAGE', 'dMRI', 'ihMTRAGE', 'B1+')
        if pattern_key == 'G-ihMTR':
            return ('dMRI', 'ihMTRAGE')
    if group == 'MESE':
        return ('MESE',)
    if group == 'MEGRE':
        if pattern_key.startswith("R2'-"):
            return ('MEGRE', 'MESE')
        return ('MEGRE',)
    if group == 'Q-Ratio':
        if pattern_key.endswith('-B1c'):
            return ('MP2RAGE', 'MEGRE', 'B1+')
        return ('MP2RAGE', 'MEGRE')
    if group == 'QSM':
        if pattern_key == 'QSM-SEPIA-E5-X' or pattern_key.endswith('R2pnet-E5-X'):
            return ('MEGRE',)
        if "R2'" in pattern_key:
            return ('MEGRE', 'MESE')
        return ('MEGRE',)
    return ()


def is_noddi_pattern(group: str, pattern_key: str) -> bool:
    return group == 'dMRI' and (
        pattern_key.startswith('ICVF')
        or pattern_key.startswith('ISOVF')
        or pattern_key.startswith('OD')
    )


def is_gm_noddi_pattern(group: str, pattern_key: str) -> bool:
    return is_noddi_pattern(group, pattern_key) and '(GM' in pattern_key


def noddi_hybrid_label(pattern_key: str) -> str:
    return display_label(pattern_key.replace(' (GM; ', ' (').replace(' (GM)', ''))


def tissues_for(group: str, pattern_key: str) -> tuple[str, ...]:
    if group == 'G-Ratio':
        return ('wm',)
    if is_noddi_pattern(group, pattern_key):
        if is_gm_noddi_pattern(group, pattern_key):
            return ('gm',)
        return ('wm', 'gmwm')
    return ('gm', 'wm', 'gmwm')


def build_metric_specs(
    patterns_file: Path,
) -> list[MetricSpec]:
    document = _load_document(patterns_file)
    if 'metrics' in document:
        return [
            MetricSpec(
                label=record['label'],
                primary_label=record['primary_label'],
                pattern_key=record['pattern_key'],
                group=record['group'],
                family=record['family'],
                source_image=record['source_image'],
                qc_modalities=tuple(record.get('qc_modalities', ())),
                tissues=tuple(record.get('tissues', ())),
                primary=bool(record.get('primary', False)),
                primary_order=record.get('primary_order'),
                expanded_order=record.get('expanded_order'),
            )
            for record in sorted(
                document['metrics'],
                key=lambda item: item.get('expanded_order', 10**9),
            )
        ]

    nested = load_patterns(patterns_file)
    specs: list[MetricSpec] = []
    primary_keys = set(PRIMARY_PATTERN_KEYS.values())

    for group, group_patterns in nested.items():
        for pattern_key in group_patterns:
            specs.append(
                MetricSpec(
                    label=display_label(pattern_key),
                    primary_label=primary_label(pattern_key),
                    pattern_key=pattern_key,
                    group=group,
                    family=infer_family(group, pattern_key),
                    source_image=source_image_from_group(group),
                    qc_modalities=qc_modalities_for(group, pattern_key),
                    tissues=tissues_for(group, pattern_key),
                    primary=pattern_key in primary_keys,
                    primary_order=(
                        PRIMARY_METRIC_LABELS.index(primary_label(pattern_key)) + 1
                        if primary_label(pattern_key) in PRIMARY_METRIC_LABELS
                        else None
                    ),
                    expanded_order=len(specs) + 1,
                )
            )
    return specs


def primary_metric_specs(
    specs: list[MetricSpec],
    tissue: str | None = None,
) -> list[MetricSpec]:
    """Return one tissue-appropriate specification per primary metric.

    Some primary metrics have distinct source maps by tissue. For example,
    cortical-GM ICVF comes from the GM-NODDI fit even though the canonical
    primary ICVF record is the WM-compatible fit. Select the first eligible
    record for each primary display label, matching the manuscript workflow.
    """
    candidates = [spec for spec in specs if tissue is None or tissue in spec.tissues]
    by_label: dict[str, MetricSpec] = {}
    for spec in candidates:
        by_label.setdefault(spec.primary_label, spec)
    return [by_label[label] for label in PRIMARY_METRIC_LABELS if label in by_label]


def metric_specs_for_analysis(
    specs: list[MetricSpec],
    analysis_set: str,
    tissue: str | None = None,
) -> list[MetricSpec]:
    candidates = [spec for spec in specs if tissue is None or tissue in spec.tissues]
    if analysis_set == 'primary':
        return primary_metric_specs(specs, tissue=tissue)
    if analysis_set in {'full', 'expanded'}:
        return candidates
    raise ValueError(f'Unsupported metric set: {analysis_set}')


def selected_analysis_sets(analysis_set: str) -> tuple[str, ...]:
    """Expand an analysis-set selector into the concrete sets to write."""

    if analysis_set == 'full':
        return ('primary', 'full')
    if analysis_set == 'primary':
        return ('primary',)
    raise ValueError(f'Unsupported metric set: {analysis_set}')


def metric_order(
    specs: list[MetricSpec],
    analysis_set: str,
    tissue: str | None = None,
) -> list[str]:
    return [
        spec.label
        for spec in metric_specs_for_analysis(
            specs,
            analysis_set,
            tissue=tissue,
        )
    ]


def metric_display_labels(
    specs: list[MetricSpec],
    analysis_set: str,
    tissue: str | None = None,
) -> dict[str, str]:
    def label_for_spec(spec: MetricSpec) -> str:
        if tissue == 'gm' and spec.group == 'dMRI' and '(GM' in spec.pattern_key:
            return spec.label.replace(' (GM; ', ' (').replace(' (GM)', '')
        return spec.label

    if analysis_set == 'primary':
        return {
            spec.label: metric_plot_label(spec.primary_label)
            for spec in primary_metric_specs(
                specs,
                tissue=tissue,
            )
        }
    if analysis_set in {'full', 'expanded'}:
        return {
            spec.label: metric_plot_label(label_for_spec(spec))
            for spec in metric_specs_for_analysis(
                specs,
                analysis_set,
                tissue=tissue,
            )
        }
    raise ValueError(f'Unsupported metric set: {analysis_set}')


def gm_noddi_hybrid_pairs(specs: list[MetricSpec]) -> dict[str, str]:
    """Map regular NODDI labels to GM-NODDI labels for GM+WM hybrids."""

    wm_by_label: dict[str, str] = {}
    gm_by_label: dict[str, str] = {}

    for spec in specs:
        if not is_noddi_pattern(spec.group, spec.pattern_key):
            continue
        hybrid_label = noddi_hybrid_label(spec.pattern_key)
        if is_gm_noddi_pattern(spec.group, spec.pattern_key):
            gm_by_label[hybrid_label] = spec.label
        else:
            wm_by_label[hybrid_label] = spec.label

    return {
        wm_label: gm_by_label[hybrid_label]
        for hybrid_label, wm_label in wm_by_label.items()
        if hybrid_label in gm_by_label
    }
