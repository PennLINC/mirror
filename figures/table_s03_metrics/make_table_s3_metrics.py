#!/usr/bin/env python3
"""Write a styled HTML table for the supplemental scalar-metric list."""

from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.metrics import (  # noqa: E402
    SOURCE_IMAGE_COLORS,
    MetricSpec,
    build_metric_specs,
    metric_plot_label,
)
from utils.paths import CODE_ROOT, RUN_FIGURES_ROOT  # noqa: E402


DIFFUSION_MEASURES = {
    'AD': 'Axial diffusivity',
    'ADE': 'Axial diffusivity of the extra-cellular compartment',
    'AK': 'Axial kurtosis',
    'AWF': 'Axonal water fraction',
    'AxonalD': 'Axonal diffusivity',
    'FA': 'Fractional anisotropy',
    'GFA': 'Generalized fractional anisotropy',
    'ISO': 'Isotropic diffusion',
    'KFA': 'Kurtosis fractional anisotropy',
    'LI': 'LI',
    'MD': 'Mean diffusivity',
    'MK': 'Mean kurtosis',
    'MKT': 'Mean of the kurtosis tensor',
    'QA': 'Quantitative anisotropy',
    'RD': 'Radial diffusivity',
    'RDE': 'Radial diffusivity of the extra-cellular compartment',
    'RK': 'Radial kurtosis',
    'Tortuosity': 'Tortuosity',
}

MAPMRI_MEASURES = {
    'NG': 'Non-Gaussianity',
    'NG Parallel': 'Parallel non-Gaussianity',
    'NG Perpendicular': 'Perpendicular non-Gaussianity',
    'PA': 'PA',
    'PAth': 'PAth',
    'RTAP': 'Return to axis probability',
    'RTOP': 'Return to origin probability',
    'RTPP': 'Return to plane probability',
}

NODDI_MEASURES = {
    'ICVF': 'Intracellular volume fraction',
    'ISOVF': 'Isotropic volume fraction',
    'OD': 'Orientation dispersion index',
}

IHMT_DESCRIPTIONS = {
    'ihMTw': 'Inhomogeneous magnetization transfer-weighted map estimated from ihMTRAGE.',
    'ihMTR': 'Inhomogeneous magnetization transfer ratio map estimated from ihMTRAGE.',
    'MTR': 'Magnetization transfer ratio map estimated from ihMTRAGE.',
    'ihMTsat': 'Inhomogeneous magnetization transfer saturation map estimated from ihMTRAGE.',
    'ihMTsat-B1c': (
        'Inhomogeneous magnetization transfer saturation map normalized by squared B1+ '
        'estimated from ihMTRAGE and B1+ field map.'
    ),
}

FAMILY_LABELS = {
    'dMRI': 'dMRI',
    'G-Ratio': 'G-Ratio',
    'ihMT': 'ihMT',
    'MEGRE': 'MEGRE',
    'MESE': 'MESE',
    'MESE/MEGRE': 'MESE/MEGRE',
    'MP2RAGE': 'MP2RAGE',
    'Q-Ratio': 'Q-Ratio',
    'QSM': 'QSM',
    'T1w/T2w': 'T₁w/T₂w',
}

FAMILY_COLORS = {
    'dMRI': SOURCE_IMAGE_COLORS['dMRI'],
    'G-Ratio': SOURCE_IMAGE_COLORS['g-ratio'],
    'ihMT': SOURCE_IMAGE_COLORS['ihMT'],
    'MEGRE': SOURCE_IMAGE_COLORS['MEGRE'],
    'MESE': SOURCE_IMAGE_COLORS['MESE'],
    'MESE/MEGRE': '#5F64B4',
    'MP2RAGE': SOURCE_IMAGE_COLORS['R1'],
    'Q-Ratio': '#9D6FBA',
    'QSM': SOURCE_IMAGE_COLORS['QSM'],
    'T1w/T2w': SOURCE_IMAGE_COLORS['T1w/T2w'],
}


@dataclass(frozen=True)
class MetricRow:
    metric: str
    family: str
    description: str


def html_style(**properties: str) -> str:
    return '; '.join(f'{name.replace("_", "-")}: {value}' for name, value in properties.items())


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    token = color.strip().lstrip('#')
    if len(token) != 6:
        return (153, 153, 153)
    return tuple(int(token[index : index + 2], 16) for index in range(0, 6, 2))


def tinted_background(color: str, alpha: float = 0.10) -> str:
    red, green, blue = hex_to_rgb(color)
    tint = tuple(round(255 * (1 - alpha) + channel * alpha) for channel in (red, green, blue))
    return f'rgb({tint[0]}, {tint[1]}, {tint[2]})'


def publication_label(spec: MetricSpec) -> str:
    key = spec.pattern_key
    if spec.group == 'T1w/T2w Ratio':
        return key.replace('-MyelinW', '-T1w/T2w')
    if spec.group == 'Q-Ratio':
        return spec.label.replace('Q-Ratio', 'q-Ratio', 1)
    if spec.group == 'dMRI':
        if key == 'DKI FA':
            return 'DKI Tensor FA'
        if key.startswith('GQI '):
            return f'DSI Studio {key}'
        dsi_match = re.fullmatch(r'([A-Z]+) \(DSIStudio\)', key)
        if dsi_match:
            return f'DSI Studio Tensor {dsi_match.group(1)}'
        noddi_match = re.fullmatch(r'(ICVF|ISOVF|OD)( \(GM\))?', key)
        if noddi_match:
            return f'NODDI {noddi_match.group(1)}{noddi_match.group(2) or ""}'
        noddi_mod_match = re.fullmatch(r'(ICVF|OD) \((GM; )?Modulated\)', key)
        if noddi_mod_match:
            gm_suffix = ' (GM)' if noddi_mod_match.group(2) else ''
            return f'NODDI {noddi_mod_match.group(1)} Modulated{gm_suffix}'
        if key in MAPMRI_MEASURES:
            suffix = {
                'NG Parallel': 'NG Par',
                'NG Perpendicular': 'NG Perp',
                'PAth': 'PATH',
            }.get(key, key)
            return f'TORTOISE MAPMRI {suffix}'
        tortoise_match = re.fullmatch(r'([A-Z]+) \(TORTOISE; (Inner|Full) Shells\)', key)
        if tortoise_match:
            model = 'MAPMRI' if tortoise_match.group(2) == 'Inner' else 'Tensor'
            return f'TORTOISE {model} {tortoise_match.group(1)}'
    return metric_plot_label(spec.label)


def publication_family(spec: MetricSpec) -> str:
    if spec.group == 'T1w/T2w Ratio':
        return 'T1w/T2w'
    if spec.group == 'MP2RAGE':
        return 'MP2RAGE'
    if spec.group == 'G-Ratio':
        return 'g-Ratio'
    if spec.group == 'Q-Ratio':
        return 'q-Ratio'
    if spec.group == 'MEGRE' and spec.pattern_key.startswith("R2'-"):
        return 'MESE/MEGRE'
    return spec.group


def dki_description(key: str) -> str:
    if key.startswith('DKI Micro '):
        measure = key.removeprefix('DKI Micro ')
        return f'{DIFFUSION_MEASURES[measure]} estimated by DIPY’s DKI microstructural model.'
    if key == 'DKI FA':
        return 'Fractional anisotropy estimated by DIPY’s DKI model.'
    if key.startswith('DKI '):
        measure = key.removeprefix('DKI ')
        return f'{DIFFUSION_MEASURES[measure]} estimated by DIPY’s DKI model.'
    raise ValueError(key)


def noddi_description(key: str) -> str:
    gm = '(GM' in key
    model = 'AMICO’s gray matter NODDI model' if gm else 'AMICO’s NODDI model'
    base = re.sub(r' \(.*\)', '', key)
    measure = NODDI_MEASURES[base]
    if 'Modulated' in key:
        return f'{measure} modulated by tissue fraction estimated by {model}.'
    return f'{measure} estimated by {model}.'


def dmri_description(key: str) -> str:
    if key.startswith('DKI '):
        return dki_description(key)
    if key.startswith('GQI '):
        measure = key.removeprefix('GQI ')
        return f'{DIFFUSION_MEASURES[measure]} estimated by DSI Studio’s GQI fit.'
    dsi_match = re.fullmatch(r'([A-Z]+) \(DSIStudio\)', key)
    if dsi_match:
        measure = dsi_match.group(1)
        return f'{DIFFUSION_MEASURES[measure]} estimated by DSI Studio’s tensor fit.'
    if key.startswith(('ICVF', 'ISOVF', 'OD')):
        return noddi_description(key)
    if key in MAPMRI_MEASURES:
        return f'{MAPMRI_MEASURES[key]} estimated by TORTOISE’s MAPMRI model.'
    tortoise_match = re.fullmatch(r'([A-Z]+) \(TORTOISE; (Inner|Full) Shells\)', key)
    if tortoise_match:
        measure, shells = tortoise_match.groups()
        model = 'MAPMRI' if shells == 'Inner' else 'tensor'
        shell_text = 'inner shells only' if shells == 'Inner' else 'all shells'
        return f'{DIFFUSION_MEASURES[measure]} estimated by TORTOISE’s {model} model from {shell_text}.'
    raise ValueError(f'No dMRI description rule for {key}')


def qsm_description(key: str) -> str:
    echoes = 'all five echoes' if '-E5-' in key else 'the last four echoes'
    if key.startswith('QSM-SEPIA-'):
        return f'Chi map from SEPIA toolbox, using {echoes} of the MEGRE scan.'

    if "-R2'-" in key:
        method = 'R2’ map estimated from MEGRE and MESE scans'
    elif '-R2pnet-' in key:
        method = 'R2pnet algorithm'
    elif '-R2*-' in key:
        method = 'R2* map estimated from MEGRE scan'
    else:
        method = 'the MEGRE scan'

    if key.endswith('-X'):
        quantity = 'Chi map'
    elif key.endswith('-Para'):
        quantity = 'Paramagnetic susceptibility map'
    elif key.endswith('-Dia'):
        quantity = 'Diamagnetic susceptibility map'
    else:
        raise ValueError(f'No QSM output-type rule for {key}')
    return f'{quantity} from χ-sep-net toolbox, using {echoes} of the MEGRE scan and {method}.'


def description(spec: MetricSpec) -> str:
    key = spec.pattern_key
    if spec.group == 'T1w/T2w Ratio':
        source = 'MPRAGE T1w' if key.startswith('MPRAGE') else 'SPACE T1w'
        return f'T1w/T2w ratio map estimated from {source} and SPACE T2w.'
    if spec.group == 'MP2RAGE':
        correction = 'with B1-correction' if key == 'R1-B1c' else 'without B1-correction'
        return f'R1 map estimated from MP2RAGE, {correction}. In s-1.'
    if spec.group == 'ihMT':
        return IHMT_DESCRIPTIONS[key]
    if spec.group == 'dMRI':
        return dmri_description(key)
    if spec.group == 'MESE':
        return 'Irreversible transverse relaxation rate, in seconds-1.'
    if spec.group == 'MEGRE':
        echoes = 'all five echoes' if key.endswith('E5') else 'the last four echoes'
        if key.startswith("R2'-"):
            return (
                f'Reversible transverse relaxation rate, in seconds-1, estimated from {echoes} '
                'of the MEGRE scan and the MESE scan.'
            )
        return f'Apparent transverse relaxation rate, in seconds-1, estimated from {echoes} of the MEGRE scan.'
    if spec.group == 'Q-Ratio':
        echoes = 'all five echoes' if '-E5' in key else 'the last four echoes'
        correction = 'B1-corrected' if key.endswith('-B1c') else 'uncorrected'
        return (
            f'R1 x R2*, estimated from the {correction} R1 derived from the MP2RAGE scan '
            f'and the R2* derived from {echoes} of the MEGRE scan. In seconds-2.'
        )
    if spec.group == 'QSM':
        return qsm_description(key)
    if spec.group == 'G-Ratio':
        if key == 'G-ihMTsat':
            return 'g-ratio map estimated from ihMTsat-B1c, ISOVF, and ICVF maps.'
        if key == 'G-ihMTR':
            return 'g-ratio map estimated from ihMTR, ISOVF, and ICVF maps.'
    raise ValueError(f'No description rule for {spec.group}: {key}')


def build_rows(patterns_file: Path) -> list[MetricRow]:
    return [
        MetricRow(
            metric=publication_label(spec),
            family=publication_family(spec),
            description=description(spec),
        )
        for spec in build_metric_specs(patterns_file)
    ]


def family_order(rows: list[MetricRow]) -> list[str]:
    observed = []
    for row in rows:
        if row.family not in observed:
            observed.append(row.family)
    return observed


def family_count(rows: list[MetricRow], family: str) -> int:
    return sum(row.family == family for row in rows)


def legend_html(families: list[str]) -> str:
    items = []
    for family in families:
        color = FAMILY_COLORS.get(family, SOURCE_IMAGE_COLORS['Other'])
        label = FAMILY_LABELS.get(family, family)
        items.append(
            '<span class="legend-item">'
            f'<span class="legend-swatch" style="background:{html.escape(color)}"></span>'
            f'{html.escape(label)}'
            '</span>'
        )
    return '\n'.join(items)


def rows_html(rows: list[MetricRow], families: list[str]) -> str:
    output = []
    for family in families:
        color = FAMILY_COLORS.get(family, SOURCE_IMAGE_COLORS['Other'])
        background = tinted_background(color)
        label = FAMILY_LABELS.get(family, family)
        count = family_count(rows, family)
        header_style = html_style(
            background_color=background,
            border_left=f'0.42rem solid {color}',
        )
        output.append(
            f'<tr class="family-row" style="{header_style}">'
            f'<th colspan="2"><span>{html.escape(label)}</span>'
            f'<span class="family-count">{count} metric{"s" if count != 1 else ""}</span></th>'
            '</tr>'
        )
        for row in [candidate for candidate in rows if candidate.family == family]:
            row_style = html_style(
                background_color=background,
                border_left=f'0.42rem solid {color}',
            )
            output.append(
                f'<tr class="metric-row" style="{row_style}">'
                f'<td class="metric-name">{html.escape(row.metric)}</td>'
                f'<td class="metric-description">{html.escape(row.description)}</td>'
                '</tr>'
            )
    return '\n'.join(output)


def write_html_table(rows: list[MetricRow], output: Path, title: str) -> None:
    families = family_order(rows)
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{
    color-scheme: light;
    font-family: Arial, Helvetica, sans-serif;
    font-size: 16px;
  }}
  body {{
    margin: 0;
    color: #222222;
    background: #ffffff;
  }}
  .table-wrap {{
    max-width: 1120px;
    margin: 0 auto;
    padding: 1.25rem;
  }}
  h1 {{
    margin: 0 0 0.35rem;
    font-size: 1.35rem;
    line-height: 1.15;
  }}
  .subtitle {{
    margin: 0 0 1rem;
    color: #555555;
    font-size: 0.95rem;
    line-height: 1.4;
  }}
  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem 0.9rem;
    align-items: center;
    margin: 0 0 0.9rem;
    font-size: 0.9rem;
    line-height: 1.2;
  }}
  .legend-title {{
    font-weight: 700;
    margin-right: 0.1rem;
  }}
  .legend-item {{
    display: inline-flex;
    align-items: center;
    gap: 0.32rem;
    white-space: nowrap;
  }}
  .legend-swatch {{
    display: inline-block;
    width: 0.82rem;
    height: 0.82rem;
    border-radius: 999px;
  }}
  table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    box-shadow: 0 0 0 1px #d7d7d7;
    border-radius: 0.6rem;
    overflow: hidden;
  }}
  thead th {{
    position: sticky;
    top: 0;
    z-index: 2;
    background: #242424;
    color: #ffffff;
    border-bottom: 2px solid #111111;
    padding: 0.62rem 0.78rem;
    text-align: left;
    font-size: 0.92rem;
    letter-spacing: 0.01em;
  }}
  .family-row th {{
    padding: 0.48rem 0.72rem;
    border-top: 1px solid #d1d1d1;
    border-bottom: 1px solid #d1d1d1;
    text-align: left;
    font-size: 0.94rem;
    letter-spacing: 0.01em;
  }}
  .family-count {{
    margin-left: 0.55rem;
    color: #555555;
    font-size: 0.82rem;
    font-weight: 600;
  }}
  .metric-row td {{
    padding: 0.48rem 0.72rem;
    border-bottom: 1px solid rgba(0, 0, 0, 0.10);
    vertical-align: top;
    line-height: 1.28;
  }}
  .metric-name {{
    width: 25%;
    min-width: 13rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }}
  .metric-description {{
    width: 75%;
  }}
  @media print {{
    .table-wrap {{
      max-width: none;
      padding: 0;
    }}
    thead th {{
      position: static;
    }}
    table {{
      box-shadow: none;
      border: 1px solid #d7d7d7;
    }}
  }}
</style>
</head>
<body>
<main class="table-wrap">
  <h1>{html.escape(title)}</h1>
  <p class="subtitle">Rows are generated from <code>configuration/metrics.yml</code> and colored by metric family.</p>
  <div class="legend"><span class="legend-title">Metric family</span>{legend_html(families)}</div>
  <table aria-label="{html.escape(title)}">
    <thead>
      <tr>
        <th>Metric</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
{rows_html(rows, families)}
    </tbody>
  </table>
</main>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    print(f'Wrote: {output}', flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--patterns-file',
        type=Path,
        default=CODE_ROOT / 'configuration' / 'metrics.yml',
        help='Metric pattern registry used as the source of truth for table rows.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=RUN_FIGURES_ROOT / 'table_s03_metrics' / 'TableS3.html',
        help='Output HTML path.',
    )
    parser.add_argument(
        '--title',
        default='Supplemental scalar MRI measures',
        help='HTML table title.',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = build_rows(args.patterns_file.expanduser().resolve())
    if len(rows) != 93:
        raise RuntimeError(f'Expected 93 supplemental metrics from metrics.yml, found {len(rows)}.')
    write_html_table(rows, args.output.expanduser().resolve(), args.title)


if __name__ == '__main__':
    main()
