"""Shared declaration of the MIRROR processing workflow, plus drawing primitives.

Figure S1 renders this graph, so the node/edge declaration lives here rather
than being embedded in the figure entry point.

Stage names and the edges between stages are not recorded in any JSON, so they
are declared below and traced from the ``processing/process_*.py`` scripts.
Scalar output labels are read from ``metrics.yml`` at import time so the
figures cannot drift from the scalar list.
"""

from __future__ import annotations

import os

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from utils.metrics import DEFAULT_METRICS_FILE, load_patterns

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Categorical slots 1-3 of the validated default palette. Three is the cap that
# clears the all-pairs colorblind floors, so the backbone stages are set apart by
# a dashed border and italic label instead of a fourth hue.
PALETTE = {
    'modality': '#2a78d6',
    'stage': '#eb6834',
    'output': '#1baf7a',
}
TINT = 0.86  # how far each fill is blended toward white
INK = '#0b0b0b'
INK_MUTED = '#52514e'
EDGE_COLOR = '#9a9894'

# Groups with more than this many scalars collapse to a single stand-in label.
STAND_IN_THRESHOLD = 5

# Acquisitions, named as in data/qc/missingness_list.tsv. MP2RAGE-P is omitted:
# process_mp2rage.py passes inv1ph/inv2ph=None, so the phase images are unused.
MODALITIES = [
    'MPRAGE T1w',
    'SPACE T1w',
    'SPACE T2w',
    'MP2RAGE',
    'B1+',
    'ihMTRAGE',
    'dMRI',
    'MESE',
    'MEGRE',
]

# Processing stages, keyed by their derivatives/ directory name.
STAGES = {
    'smriprep': ('sMRIPrep', 'anatomical reference,\nbrain mask, MNI transform'),
    'qsiprep': ('QSIPrep', 'DWI preprocessing'),
    'qsirecon': ('QSIRecon', 'diffusion model fitting'),
    'pymp2rage': ('pymp2rage', 'T₁ / R₁ mapping'),
    'ihmt': ('ihmt', 'MT saturation'),
    't1wt2w_ratio': ('t1wt2w_ratio', 'myelin-weighted ratio'),
    'mese': ('mese', 'T₂ / R₂ mapping'),
    'megre': ('megre', "R₂* and R₂' mapping"),
    'qsm': ('qsm', 'SEPIA, χ-separation'),
    'g_ratio': ('g_ratio', 'aggregate g-ratio'),
}

# sMRIPrep supplies the anatomical reference and MNI transform that every other
# stage normalizes through. Drawing ten edges for that would bury the data flow,
# so each layout renders it as a band instead.
BACKBONE = ('smriprep', 'qsiprep')
BACKBONE_NOTE = (
    'sMRIPrep supplies the T₁w reference, brain mask and MNI152NLin2009cAsym\n'
    'transform that every stage below normalizes through. It is built from the\n'
    'cross-session MPRAGE T₁w, so it is not a per-session dependency of any scalar.'
)

# Direct data flow, traced from the collect_run_data queries in processing/.
# Nodes are prefixed because 'dMRI' and 'MP2RAGE' name both a modality and a
# scalar group.
EDGES = [
    ('mod:MPRAGE T1w', 'stage:smriprep'),
    ('mod:MPRAGE T1w', 'stage:t1wt2w_ratio'),
    ('mod:SPACE T1w', 'stage:t1wt2w_ratio'),
    ('mod:SPACE T2w', 'stage:t1wt2w_ratio'),
    ('stage:t1wt2w_ratio', 'out:T1w/T2w Ratio'),
    ('mod:MP2RAGE', 'stage:pymp2rage'),
    ('mod:B1+', 'stage:pymp2rage'),
    ('stage:pymp2rage', 'out:MP2RAGE'),
    ('mod:ihMTRAGE', 'stage:ihmt'),
    ('stage:pymp2rage', 'stage:ihmt'),  # T1map and T1w-space TB1map
    ('stage:ihmt', 'out:ihMT'),
    ('mod:dMRI', 'stage:qsiprep'),
    ('stage:smriprep', 'stage:qsiprep'),
    ('stage:qsiprep', 'stage:qsirecon'),
    ('stage:qsirecon', 'out:dMRI'),
    ('mod:MESE', 'stage:mese'),
    ('stage:mese', 'stage:megre'),  # R2 map, needed for R2'
    ('mod:MEGRE', 'stage:megre'),
    ('mod:MEGRE', 'stage:qsm'),
    ('stage:megre', 'stage:qsm'),  # R2* and R2' maps
    ('stage:qsm', 'out:QSM'),
    ('stage:qsirecon', 'stage:g_ratio'),  # NODDI ISOVF and ICVF
    ('stage:ihmt', 'stage:g_ratio'),
    ('stage:g_ratio', 'out:G-Ratio'),
]

# One track per scalar group: the modalities it consumes and its stage chain, in
# processing order. Ordered so that the cross-track edges (pymp2rage -> ihmt,
# ihmt -> g_ratio, qsirecon -> g_ratio) connect neighbouring tracks and stay
# short. G-Ratio takes no modality directly; both its inputs are derivatives.
FAMILIES = [
    ('T1w/T2w Ratio', ['MPRAGE T1w', 'SPACE T1w', 'SPACE T2w'], ['t1wt2w_ratio']),
    ('MP2RAGE', ['MP2RAGE', 'B1+'], ['pymp2rage']),
    ('ihMT', ['ihMTRAGE'], ['ihmt']),
    ('G-Ratio', [], ['g_ratio']),
    ('dMRI', ['dMRI'], ['qsiprep', 'qsirecon']),
    ('QSM', ['MESE', 'MEGRE'], ['mese', 'megre', 'qsm']),
]

MAX_STAGE_DEPTH = max(len(stages) for _, _, stages in FAMILIES)

# What one stage actually hands to another, for layouts that label the hand-off
# instead of drawing a line across the whole figure.
DERIVATIVE_LABELS = {
    ('smriprep', 'qsiprep'): 'anatomical reference',
    ('pymp2rage', 'ihmt'): 'T₁map, TB₁map',
    ('mese', 'megre'): 'R₂ map',
    ('megre', 'qsm'): "R₂*, R₂'",
    ('qsirecon', 'g_ratio'): 'NODDI ISOVF, ICVF',
    ('ihmt', 'g_ratio'): 'ihMTsat, ihMTR',
}


def family_of(stage):
    """Return the scalar group whose track a stage belongs to."""
    for group, _, stages in FAMILIES:
        if stage in stages:
            return group
    return None


def crossing_inputs(group):
    """Derivative inputs a track takes from stages outside itself.

    Returns
    -------
    list of tuple
        ``(source_stage, target_stage, label)`` triples, in EDGES order.
    """
    stages = {g: s for g, _, s in FAMILIES}[group]
    found = []
    for src, dst in EDGES:
        if not (src.startswith('stage:') and dst.startswith('stage:')):
            continue
        src_stage = src.split(':', 1)[1]
        dst_stage = dst.split(':', 1)[1]
        if dst_stage in stages and src_stage not in stages:
            found.append((src_stage, dst_stage, DERIVATIVE_LABELS[(src_stage, dst_stage)]))
    return found


def modality_targets(group):
    """Map each of a track's modalities to the stage in that track it feeds.

    A modality can feed more than one stage (MEGRE goes to both megre and qsm);
    the earliest stage in the chain is the one to draw.
    """
    stages = {g: s for g, _, s in FAMILIES}[group]
    targets = {}
    for src, dst in EDGES:
        if not (src.startswith('mod:') and dst.startswith('stage:')):
            continue
        modality = src.split(':', 1)[1]
        stage = dst.split(':', 1)[1]
        if stage not in stages:
            continue
        if modality not in targets or stages.index(stage) < stages.index(targets[modality]):
            targets[modality] = stage
    return targets


def load_output_groups():
    """Build the scalar output labels from metrics.yml.

    Returns
    -------
    dict
        Maps scalar group name to ``(headline, detail)``. Groups with more than
        ``STAND_IN_THRESHOLD`` scalars collapse to a stand-in name and a count;
        smaller groups list their scalar keys.
    """
    patterns = load_patterns(DEFAULT_METRICS_FILE)

    declared = {name.split(':', 1)[1] for name in _nodes_of_kind('out')}
    if declared != set(patterns):
        raise ValueError(
            f'Workflow output nodes {sorted(declared)} do not match the scalar '
            f'groups in metrics.yml {sorted(patterns)}'
        )

    groups = {}
    for group, scalars in patterns.items():
        if len(scalars) > STAND_IN_THRESHOLD:
            groups[group] = (f'{group} Scalars', f'{len(scalars)} maps')
        else:
            groups[group] = (group, '\n'.join(scalars))

    return groups


def _nodes_of_kind(kind):
    """Collect node IDs of one kind, in the order they first appear in EDGES."""
    seen = []
    for src, dst in EDGES:
        for node in (src, dst):
            if node.startswith(f'{kind}:') and node not in seen:
                seen.append(node)
    return seen


def node_label(node, output_groups):
    """Return ``(headline, detail)`` for a node ID."""
    kind, name = node.split(':', 1)
    if kind == 'mod':
        return name, ''
    if kind == 'stage':
        return STAGES[name]
    return output_groups[name]


def node_size(node, output_groups, base_width=17.0):
    """Return ``(width, height)`` for a node, grown to fit its detail lines."""
    _, detail = node_label(node, output_groups)
    n_lines = len(detail.splitlines()) if detail else 0
    kind = node.split(':', 1)[0]
    width = {'mod': base_width * 0.88, 'stage': base_width, 'out': base_width * 1.25}[kind]
    return width, max(6.4, 4.6 + 2.1 * n_lines)


def tint(hex_color, amount=TINT):
    """Blend a hex color toward white; used for node fills."""
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    r, g, b = (c + (1 - c) * amount for c in (r, g, b))
    return (r, g, b)


def draw_node(ax, x, y, width, height, node, output_groups, fontsize=9):
    """Draw one rounded node box centered on (x, y)."""
    kind = node.split(':', 1)[0]
    role = {'mod': 'modality', 'stage': 'stage', 'out': 'output'}[kind]
    is_backbone = kind == 'stage' and node.split(':', 1)[1] in BACKBONE
    headline, detail = node_label(node, output_groups)
    draw_box(ax, x, y, width, height, headline, detail, role, is_backbone, fontsize)


def draw_box(
    ax,
    x,
    y,
    width,
    height,
    headline,
    detail,
    role,
    dashed=False,
    fontsize=9,
    color=None,
    facecolor=None,
):
    """Draw one rounded box with a bold headline and a muted detail line."""
    color = PALETTE[role] if color is None else color
    facecolor = tint(color) if facecolor is None else facecolor

    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle='round,pad=0,rounding_size=0.35',
            facecolor=facecolor,
            edgecolor=color,
            linewidth=1.6,
            # Secondary encoding for the backbone stages, so identity never
            # rests on color alone.
            linestyle=(0, (4, 2)) if dashed else 'solid',
            zorder=3,
        )
    )

    if detail:
        headline_lines = headline.count('\n') + 1
        if headline_lines == 1:
            headline_y = y + height * 0.23
            detail_y = y - height * 0.22
        else:
            headline_y = y + height * 0.12
            detail_y = y - height * 0.28
        ax.text(
            x,
            headline_y,
            headline,
            ha='center',
            va='center',
            fontsize=fontsize,
            fontweight='bold',
            fontstyle='italic' if dashed else 'normal',
            color=INK,
            zorder=4,
        )
        ax.text(
            x,
            detail_y,
            detail,
            ha='center',
            va='center',
            fontsize=fontsize - 2,
            color=INK_MUTED,
            linespacing=1.35,
            zorder=4,
        )
    else:
        ax.text(
            x,
            y,
            headline,
            ha='center',
            va='center',
            fontsize=fontsize,
            fontweight='bold',
            color=INK,
            zorder=4,
        )


def draw_edge(
    ax,
    start,
    end,
    rad=0.08,
    color=EDGE_COLOR,
    linewidth=1.2,
    head=(5, 3),
    zorder=2,
    linestyle='solid',
):
    """Draw one curved arrow between two anchor points."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            connectionstyle=f'arc3,rad={rad}',
            arrowstyle=f'-|>,head_length={head[0]},head_width={head[1]}',
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def add_legend(ax, x, y, fontsize=9, vertical=True):
    """Draw the three-tier key plus the backbone marker."""
    entries = [
        ('modality', 'Input modality', 'solid'),
        ('stage', 'Processing stage', 'solid'),
        ('stage', 'Backbone stage (cross-session)', (0, (4, 2))),
        ('output', 'Scalar outputs', 'solid'),
    ]
    step = -2.6 if vertical else 0
    for i, (role, label, style) in enumerate(entries):
        color = PALETTE[role]
        cy = y + i * step
        cx = x if vertical else x + i * 26
        ax.add_patch(
            FancyBboxPatch(
                (cx, cy - 0.7),
                2.6,
                1.6,
                boxstyle='round,pad=0,rounding_size=0.3',
                facecolor=tint(color),
                edgecolor=color,
                linewidth=1.5,
                linestyle=style,
                zorder=3,
            )
        )
        ax.text(
            cx + 3.4,
            cy + 0.1,
            label,
            ha='left',
            va='center',
            fontsize=fontsize,
            color=INK_MUTED,
            zorder=4,
        )


def save(fig, name):
    """Write a figure to figures/<name>.png and .pdf."""
    out_dir = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'figures'))
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(
            os.path.join(out_dir, f'{name}.{ext}'),
            bbox_inches='tight',
            dpi=400 if ext == 'png' else None,
            facecolor='white',
        )
    print(f'Wrote {os.path.join(out_dir, name)}.png / .pdf')
