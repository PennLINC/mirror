"""Plot the layered scalar dependency graph, intermediate derivatives included.

This is the layered counterpart to plot_workflow_modality_sets.py. That figure
reads scalar_modalities.json, where every scalar is attributed straight back to
the acquisitions it ultimately comes from, so a scalar set is one hop from its
modalities. Here the source is scalar_modalities_layered.json, where a scalar may
depend on other scalars: R2' comes from R2* and R2, the g-ratios come from ihMT
and NODDI, and chi-separation with a measured R2' comes from MEGRE and R2'.

A scalar something else depends on is drawn as its own node, so the reused
derivatives are visible instead of being folded into their consumers. Scalars
nothing depends on are still grouped by their exact dependency signature, which
keeps the 53 dMRI maps and the 14 R2*-only QSM maps to one box each. Two things
override that: scalars in KEEP_GROUPED stay inside their group even though
something depends on them, and MERGES draws distinct scalars as one node.

Columns are dependency depth: source modalities, then scalars one hop out, then
two, then three. The layout is the standard three-step layered drawing, so
nothing below the column assignment is hand-placed:

1. Any edge spanning more than one column is split into single-column hops
   around waypoint nodes (``build_routing``). A waypoint takes a real slot in
   each column it crosses, which is what keeps MEGRE -> chi-separation from
   cutting through the level 1 and 2 boxes on its way to level 3.
2. Column order is chosen to minimise edge crossings (``minimise_crossings``),
   by repeated median sweeps plus adjacent-swap refinement. Only MODALITY_ORDER
   is held fixed, as the reading order for the figure.
3. Heights are relaxed toward the mean of each node's neighbours without
   disturbing that order (``place_column``), so edges stay short.

Every edge is then routed the same way (``edge_polyline``): flat across each
column, at the height of the box it leaves, the waypoint it was given, or the box
it lands on, and turning only in the gutters in between, where there is nothing
to hit. Because each turn leaves and arrives level, a segment can never bow
outside the span of its own two ends, so no edge can cross a box it does not
belong to. ``check_no_box_intrusions`` asserts exactly that at draw time.

MPRAGE T1w is the sMRIPrep anatomical reference, so it feeds everything. Drawing
that to every scalar node would bury the data flow, so the pale backbone edges
stop at the first scalar layer and the legend covers the rest.
"""

import json
import os
import re
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path

from workflow_graph import (
    EDGE_COLOR,
    INK_MUTED,
    MODALITIES,
    PALETTE,
    draw_box,
    draw_edge,
    save,
    tint,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Every scalar map is normalized through the sMRIPrep anatomical reference,
# which is built from this modality.
UNIVERSAL = 'MPRAGE T1w'

# Left-column order, top to bottom. The scalar columns order themselves against
# this, so it is the one piece of layout that stays hand-tuned.
MODALITY_ORDER = [
    'MPRAGE T1w',
    'SPACE T1w',
    'SPACE T2w',
    'MP2RAGE',
    'B1+',
    'ihMTRAGE',
    'dMRI',
    'MEGRE',
    'MESE',
]

# Scalars that differ only in how many MEGRE echoes were fit are the same
# contrast measured two ways, so they collapse to one node. The suffix is
# stripped from dependencies too, which is what merges R2*-E4/R2*-E5 into the
# single R2* that R2' and the Q-ratios point at.
ECHO_SUFFIX = re.compile(r'-E\d+')

# Scalars drawn as one node despite having different inputs. The node takes the
# union of their inputs, so the g-ratio box shows both the ihMTR and the
# ihMTsat-B1c route into it.
MERGES = {
    'G-ihMTsat': 'G-Ratio',
    'G-ihMTR': 'G-Ratio',
}

# Reused scalars that stay inside their group instead of being promoted to their
# own node. The g-ratios take NODDI ICVF and ISOVF, but pulling two maps out of
# the 53-map dMRI box says less than pointing the edge at the box itself.
KEEP_GROUPED = ('NODDI ICVF', 'NODDI ISOVF')

# Groups too large to name scalar by scalar get a stand-in and a qualifier, keyed
# by (depth, signature). The QSM groups share a stand-in name, so the qualifier
# and the signature underneath are what tell them apart.
STAND_INS = {
    (1, ('dMRI',)): ('dMRI Scalars', 'DKI, DSI Studio, NODDI, TORTOISE'),
    (1, ('MEGRE',)): ('QSM Scalars', 'SEPIA and χ-separation without a measured R₂′'),
    (3, ('MEGRE', "R2'")): ('QSM Scalars', 'χ-separation using a measured R₂′'),
}

MAX_NAMED = 3

# Characters per line before a dependency signature wraps.
WRAP_AT = 30

COLUMN_HEADERS = {
    0: 'SOURCE MODALITY',
    1: 'LEVEL 1 SCALARS',
    2: 'LEVEL 2 SCALARS',
    3: 'LEVEL 3 SCALARS',
}

MOD_W = 26.0
MOD_H = 13.0

# Scalar boxes are sized to their text rather than fixed, so a one-word
# derivative stays a small pill and only the grouped sets get wide.
MIN_W = 23.0
MAX_W = 58.0
MIN_H = 9.0
LINE_H = 2.9
PAD_H = 3.4
CHAR_W_HEAD = 2.6
CHAR_W_DETAIL = 2.05

COL_GAP = 21.0
GAP = 2.6
TOP = 96.0

# Vertical channel reserved in a column for one edge routed past it. Wide enough
# that a skipping edge clears the boxes either side without a visible kink.
VIA_H = 3.2

# Horizontal pull at each turn, as a fraction of the gap being crossed. 0.5 is
# the classic flowchart S: the curve leaves and arrives level, and never doubles
# back on itself.
SPLINE_TENSION = 0.5

# Crossing-minimization passes. Each is a median-ordering sweep down and back up
# the layers plus adjacent-swap refinement; the best ordering seen is kept.
ORDER_PASSES = 12

# Refinement passes over the scalar columns. The first places each node under its
# inputs; later ones also pull it toward the nodes it feeds, which is what drags
# R1 down beside the Q-ratios instead of stranding it at the top.
SWEEPS = 4

# Order-preserving relaxation steps used to settle y coordinates in one column.
RELAX_PASSES = 24

FONT_MOD = 13
FONT_SET = 12
FONT_HEADER = 13
FONT_LEGEND = 11

# Data units per inch. Lower makes the rendered figure physically smaller, which
# is what makes a given point size read larger.
SCALE = 9.0

FAINT = '#d5d2cc'

# Applied longest-first so R2' and R2* are not eaten by the bare R2 rule.
_PRETTY = [
    ("R2'", 'R₂′'),
    ('R2*', 'R₂*'),
    ('T1w', 'T₁w'),
    ('T2w', 'T₂w'),
    ('B1+', 'B₁⁺'),
    ('B1c', 'B₁c'),
    ('R1', 'R₁'),
    ('R2', 'R₂'),
]


def prettify(text):
    """Replace ASCII stand-ins for subscripts and primes in a label."""
    for plain, pretty in _PRETTY:
        text = text.replace(plain, pretty)
    return text


def wrap_signature(pretty):
    """Break a ``a + b + c`` signature onto as few lines as WRAP_AT allows."""
    lines = []
    current = ''
    for part in pretty.split(' + '):
        candidate = f'{current} + {part}' if current else part
        if current and len(candidate) > WRAP_AT:
            lines.append(current)
            current = part
        else:
            current = candidate
    lines.append(current)
    return '\n'.join(lines)


def canonical(name):
    """Return the node name a scalar is drawn under."""
    stripped = ECHO_SUFFIX.sub('', name)
    return MERGES.get(stripped, stripped)


def merge_variants(deps):
    """Collapse echo variants and MERGES entries into one entry per node name.

    Returns
    -------
    tuple
        ``(merged, members)``. ``merged`` maps each canonical name to the union
        of its variants' inputs, canonicalized the same way. ``members`` maps it
        back to the original scalars it stands for, so the map counts stay true.
    """
    merged = OrderedDict()
    members = OrderedDict()
    for name, inputs in deps.items():
        key = canonical(name)
        merged.setdefault(key, [])
        for dependency in inputs:
            dependency = canonical(dependency)
            if dependency not in merged[key]:
                merged[key].append(dependency)
        members.setdefault(key, []).append(name)
    return merged, members


def load_graph():
    """Read the layered dependency map, merge variants, split out the modalities.

    Returns
    -------
    tuple
        ``(deps, members, modalities)``, where ``deps`` maps each node name to
        its inputs and ``modalities`` is the set of inputs that are not
        themselves scalars.
    """
    with open(
        os.path.join(_SCRIPT_DIR, '..', 'configuration', 'scalar_modalities_layered.json'),
        'r',
    ) as fo:
        raw = json.load(fo)

    deps, members = merge_variants(raw)

    unknown = [name for name in KEEP_GROUPED if name not in deps]
    if unknown:
        raise ValueError(f'KEEP_GROUPED names are not scalars in the file: {unknown}')

    modalities = {name for inputs in deps.values() for name in inputs} - set(deps)

    unknown = modalities - set(MODALITIES)
    if unknown:
        raise ValueError(
            'scalar_modalities_layered.json depends on names that are neither scalars in '
            f'the file nor modalities in workflow_graph.MODALITIES: {sorted(unknown)}'
        )
    if set(MODALITY_ORDER) != modalities:
        raise ValueError(
            'MODALITY_ORDER is out of date with scalar_modalities_layered.json:\n'
            f'  missing: {sorted(modalities - set(MODALITY_ORDER))}\n'
            f'  stale:   {sorted(set(MODALITY_ORDER) - modalities)}'
        )

    return deps, members, modalities


def compute_depths(deps, modalities):
    """Map every name to its longest path back to a source modality."""
    depths = {modality: 0 for modality in modalities}
    resolving = set()

    def depth(name):
        if name in depths:
            return depths[name]
        if name in resolving:
            raise ValueError(f'Dependency cycle through {name!r}')
        resolving.add(name)
        depths[name] = 1 + max(depth(dependency) for dependency in deps[name])
        resolving.discard(name)
        return depths[name]

    for name in deps:
        depth(name)

    return depths


def set_text(depth, signature, names, n_maps):
    """Return ``(headline, detail)`` for one group of terminal scalars."""
    pretty = wrap_signature(prettify(' + '.join(signature)))
    if (depth, signature) in STAND_INS:
        headline, qualifier = STAND_INS[(depth, signature)]
        return headline, f'{pretty}\n{n_maps} maps · {qualifier}'
    if len(names) > MAX_NAMED:
        return f'{n_maps} maps', pretty
    # A node standing for more maps than it names is one whose variants merged.
    if n_maps > len(names):
        pretty = f'{pretty}\n{n_maps} maps'
    return '\n'.join(prettify(name) for name in names), pretty


def build_nodes(deps, members, modalities, depths):
    """Build the drawable nodes: one per modality, per reused scalar, per group.

    A scalar something else depends on becomes its own node unless KEEP_GROUPED
    holds it back. The rest are grouped by ``(depth, dependency signature)``, so
    scalars that stand or fall together share a box. Dependencies on a scalar
    that stayed grouped are redirected to the box it ended up in.
    """
    reused = {name for inputs in deps.values() for name in inputs} & set(deps)
    reused -= set(KEEP_GROUPED)

    grouped = OrderedDict()
    for scalar, inputs in deps.items():
        if scalar in reused:
            continue
        grouped.setdefault((depths[scalar], tuple(inputs)), []).append(scalar)

    node_of = {modality: f'mod:{modality}' for modality in modalities}
    node_of.update({scalar: f'scalar:{scalar}' for scalar in reused})
    for (depth, signature), names in grouped.items():
        node_id = f'set:{depth}:{"|".join(signature)}'
        node_of.update({name: node_id for name in names})

    def resolve(node_id, inputs):
        """Map input names to node IDs, dropping duplicates and self-edges."""
        resolved = []
        for name in inputs:
            source = node_of[name]
            if source != node_id and source not in resolved:
                resolved.append(source)
        return resolved

    nodes = OrderedDict()
    for modality in MODALITY_ORDER:
        nodes[node_of[modality]] = {
            'kind': 'mod',
            'depth': 0,
            'headline': prettify(modality),
            'detail': '',
            'inputs': [],
        }

    for scalar in deps:
        if scalar not in reused:
            continue
        node_id = node_of[scalar]
        nodes[node_id] = {
            'kind': 'scalar',
            'depth': depths[scalar],
            'headline': prettify(scalar),
            'detail': '',
            'inputs': resolve(node_id, deps[scalar]),
        }

    for (depth, signature), names in grouped.items():
        node_id = node_of[names[0]]
        n_maps = sum(len(members[name]) for name in names)
        headline, detail = set_text(depth, signature, names, n_maps)
        nodes[node_id] = {
            'kind': 'set',
            'depth': depth,
            'headline': headline,
            'detail': detail,
            'inputs': resolve(node_id, signature),
        }

    return nodes


def node_size(node):
    """Return ``(width, height)`` for a node, grown to fit its own text."""
    if node['kind'] == 'mod':
        return MOD_W, MOD_H

    head = node['headline'].splitlines()
    detail = node['detail'].splitlines()
    widths = [CHAR_W_HEAD * len(line) for line in head]
    widths += [CHAR_W_DETAIL * len(line) for line in detail]
    width = min(MAX_W, max(MIN_W, max(widths) + 5.0))
    return width, max(MIN_H, PAD_H + LINE_H * (len(head) + len(detail)))


def build_routing(nodes, edges):
    """Split every edge spanning more than one column into single-column hops.

    Returns
    -------
    tuple
        ``(chains, vias)``. ``chains`` maps each original edge to the list of
        node IDs it now passes through; ``vias`` holds the waypoint nodes made
        for the columns in between. Because a waypoint takes a real slot in
        every column it crosses, the boxes there make room for it and the edge
        no longer has to pass behind them.
    """
    chains = OrderedDict()
    vias = OrderedDict()
    for src, dst in edges:
        chain = [src]
        for depth in range(nodes[src]['depth'] + 1, nodes[dst]['depth']):
            node_id = f'via:{src}->{dst}@{depth}'
            vias[node_id] = {'kind': 'via', 'depth': depth}
            chain.append(node_id)
        chain.append(dst)
        chains[(src, dst)] = chain
    return chains, vias


def count_crossings(upper, lower, succ):
    """Count crossing edge pairs between two adjacent, ordered layers."""
    rank = {node_id: i for i, node_id in enumerate(lower)}
    targets = []
    for node_id in upper:
        targets.extend(sorted(rank[v] for v in succ.get(node_id, ()) if v in rank))
    return sum(
        1
        for i in range(len(targets))
        for j in range(i + 1, len(targets))
        if targets[i] > targets[j]
    )


def total_crossings(layers, succ):
    """Count crossings over every adjacent pair of layers."""
    depths = list(layers)
    return sum(count_crossings(layers[a], layers[b], succ) for a, b in zip(depths, depths[1:]))


def median_of(node_id, adjacent, rank):
    """Weighted median position of a node's neighbours in the adjacent layer.

    Returns ``-1`` for a node with no neighbours there, which marks it as one to
    leave where it is rather than sort to the top.
    """
    positions = sorted(rank[name] for name in adjacent.get(node_id, ()) if name in rank)
    if not positions:
        return -1.0
    middle = len(positions) // 2
    if len(positions) % 2:
        return float(positions[middle])
    if len(positions) == 2:
        return (positions[0] + positions[1]) / 2
    left = positions[middle - 1] - positions[0]
    right = positions[-1] - positions[middle]
    if left + right == 0:
        return (positions[middle - 1] + positions[middle]) / 2
    return (positions[middle - 1] * right + positions[middle] * left) / (left + right)


def order_by_median(layer, adjacent, rank):
    """Reorder one layer by neighbour median, holding unattached nodes in place."""
    keys = {node_id: median_of(node_id, adjacent, rank) for node_id in layer}
    pinned = [(i, node_id) for i, node_id in enumerate(layer) if keys[node_id] < 0]
    ordered = sorted(
        (node_id for node_id in layer if keys[node_id] >= 0),
        key=lambda node_id: keys[node_id],
    )
    for i, node_id in pinned:
        ordered.insert(i, node_id)
    return ordered


def touching_crossings(layers, depth, succ):
    """Crossings on the edges that touch one layer."""
    total = 0
    if depth - 1 in layers:
        total += count_crossings(layers[depth - 1], layers[depth], succ)
    if depth + 1 in layers:
        total += count_crossings(layers[depth], layers[depth + 1], succ)
    return total


def transpose(layers, succ, fixed_depth):
    """Swap adjacent nodes for as long as doing so removes crossings."""
    improved = True
    while improved:
        improved = False
        for depth, layer in layers.items():
            if depth == fixed_depth:
                continue
            for i in range(len(layer) - 1):
                before = touching_crossings(layers, depth, succ)
                layer[i], layer[i + 1] = layer[i + 1], layer[i]
                if touching_crossings(layers, depth, succ) < before:
                    improved = True
                else:
                    layer[i], layer[i + 1] = layer[i + 1], layer[i]


def minimise_crossings(layers, succ, pred, fixed_depth=0):
    """Order every layer to minimise edge crossings, keeping ``fixed_depth`` as is.

    Alternating median sweeps set a rough order and adjacent swaps polish it. The
    ordering is only ever accepted when it beats the best seen, so the result
    cannot be worse than the natural order it started from.
    """
    snapshot = OrderedDict((depth, list(ids)) for depth, ids in layers.items())
    best = snapshot
    best_score = total_crossings(layers, succ)

    depths = list(layers)
    for step in range(ORDER_PASSES):
        downward = step % 2 == 0
        sweep = depths[1:] if downward else depths[-2::-1]
        for depth in sweep:
            if depth == fixed_depth:
                continue
            neighbour_depth = depth - 1 if downward else depth + 1
            if neighbour_depth not in layers:
                continue
            rank = {name: i for i, name in enumerate(layers[neighbour_depth])}
            layers[depth] = order_by_median(layers[depth], pred if downward else succ, rank)

        transpose(layers, succ, fixed_depth)

        score = total_crossings(layers, succ)
        if score < best_score:
            best_score = score
            best = OrderedDict((depth, list(ids)) for depth, ids in layers.items())

    return best, best_score


def place_column(order, desired, heights, bottom):
    """Assign y coordinates down one column without disturbing ``order``.

    Nodes are stacked top-down at their target heights, the stack is nudged back
    inside the figure span, then each node is relaxed toward its target within
    whatever slack its neighbours leave. Holding the order fixed is what makes
    the crossing-minimisation pass mean anything: without it, packing by target
    height would silently reorder the column again.
    """
    ys = {}
    cursor = TOP
    for node_id in order:
        height = heights[node_id]
        ys[node_id] = min(desired[node_id], cursor - height / 2)
        cursor = ys[node_id] - height / 2 - GAP

    low = min(ys[node_id] - heights[node_id] / 2 for node_id in order)
    high = max(ys[node_id] + heights[node_id] / 2 for node_id in order)
    shift = min(max(0.0, bottom - low), max(0.0, TOP - high))
    if shift:
        ys = {node_id: y + shift for node_id, y in ys.items()}

    for _ in range(RELAX_PASSES):
        for i, node_id in enumerate(order):
            half = heights[node_id] / 2
            if i:
                above = order[i - 1]
                ceiling = ys[above] - heights[above] / 2 - GAP - half
            else:
                ceiling = TOP - half
            if i < len(order) - 1:
                below = order[i + 1]
                floor_ = ys[below] + heights[below] / 2 + GAP + half
            else:
                floor_ = -np.inf
            if floor_ > ceiling:
                continue
            ys[node_id] = min(max(desired[node_id], floor_), ceiling)

    return ys


def spline_path(points, tension=SPLINE_TENSION):
    """Return a smooth cubic Bezier path through ``points``.

    Every waypoint is entered and left horizontally, which is what makes the
    routing safe rather than merely tidy: with both control points of a segment
    level with its own ends, the curve's vertical span is exactly the span
    between those two ends. It cannot bow out of the gutter it turns in, and a
    run across a column stays flat at the height reserved for it.
    """
    pts = [np.asarray(point, dtype=float) for point in points]
    # A box exactly as wide as its column puts two waypoints in the same place,
    # which would leave a segment with no length and no defined direction.
    deduped = [pts[0]]
    for point in pts[1:]:
        if not np.allclose(point, deduped[-1]):
            deduped.append(point)
    pts = deduped
    if len(pts) < 2:
        return Path(pts * 2, [Path.MOVETO, Path.LINETO])

    verts = [pts[0]]
    codes = [Path.MOVETO]
    for start, end in zip(pts, pts[1:]):
        handle = np.array([abs(end[0] - start[0]) * tension, 0.0])
        verts += [start + handle, end - handle, end]
        codes += [Path.CURVE4] * 3
    return Path(verts, codes)


def check_no_box_intrusions(routes, rects, margin=0.35, samples=800):
    """Raise if any routed edge passes through a box that is not its own endpoint.

    The routing rule makes this impossible by construction, so this is here to
    catch the case where a later change to the geometry quietly breaks it: new
    scalars in the config move the boxes, and a silently wrong figure is worse
    than a loud failure.
    """
    intrusions = []
    for (src, dst), points in routes.items():
        drawn = spline_path(points).interpolated(samples).vertices
        for node_id, (left, right, low, high) in rects.items():
            if node_id in (src, dst):
                continue
            inside = (
                (drawn[:, 0] > left + margin)
                & (drawn[:, 0] < right - margin)
                & (drawn[:, 1] > low + margin)
                & (drawn[:, 1] < high - margin)
            )
            if inside.any():
                intrusions.append(f'{src} -> {dst} crosses {node_id}')
    if intrusions:
        raise AssertionError(
            'Edge routing passes through {} box(es):\n  {}'.format(
                len(intrusions), '\n  '.join(intrusions)
            )
        )


def draw_spline(ax, points, color=EDGE_COLOR, linewidth=1.2, head=(5, 3), zorder=2):
    """Draw one arrow that follows a smooth path through ``points``."""
    ax.add_patch(
        FancyArrowPatch(
            path=spline_path(points),
            arrowstyle=f'-|>,head_length={head[0]},head_width={head[1]}',
            color=color,
            linewidth=linewidth,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


if __name__ == '__main__':
    deps, members, modalities = load_graph()
    depths = compute_depths(deps, modalities)
    nodes = build_nodes(deps, members, modalities, depths)

    edges = [(src, node_id) for node_id, node in nodes.items() for src in node['inputs']]
    chains, vias = build_routing(nodes, edges)

    sizes = {node_id: node_size(node) for node_id, node in nodes.items()}
    widths = {node_id: size[0] for node_id, size in sizes.items()}
    heights = {node_id: size[1] for node_id, size in sizes.items()}
    widths.update({node_id: 0.0 for node_id in vias})
    heights.update({node_id: VIA_H for node_id in vias})

    placed = OrderedDict(nodes)
    placed.update(vias)

    # The routing graph joins adjacent columns only, so ordering and spacing see
    # the skipping edges as well, not just their endpoints.
    segments = list(
        OrderedDict.fromkeys(
            segment for chain in chains.values() for segment in zip(chain, chain[1:])
        )
    )
    succ, pred = {}, {}
    for src, dst in segments:
        succ.setdefault(src, []).append(dst)
        pred.setdefault(dst, []).append(src)

    # Layer 0 keeps MODALITY_ORDER; the rest start in build order and get sorted.
    layers = OrderedDict()
    ordered_ids = [f'mod:{modality}' for modality in MODALITY_ORDER]
    ordered_ids += [node_id for node_id in placed if placed[node_id]['depth'] > 0]
    for node_id in ordered_ids:
        layers.setdefault(placed[node_id]['depth'], []).append(node_id)
    layers = OrderedDict(sorted(layers.items()))

    build_order = total_crossings(layers, succ)
    layers, crossings = minimise_crossings(layers, succ, pred)
    print(f'edge crossings: {crossings} ({build_order} before ordering)')

    # The tallest column sets the height of the figure; every other column is
    # placed inside that same span. Waypoints count, since they take real space.
    span = max(
        sum(heights[node_id] for node_id in ids) + GAP * (len(ids) - 1) for ids in layers.values()
    )
    bottom = TOP - span

    boxes = OrderedDict(
        (depth, [node_id for node_id in ids if node_id in nodes]) for depth, ids in layers.items()
    )

    # Each column is as wide as its widest box. The boundaries matter as much as
    # the centre: edges run horizontally out to them and only turn once past.
    col_x, col_left, col_right = {}, {}, {}
    cursor = 0.0
    for depth, ids in boxes.items():
        col_w = max(widths[node_id] for node_id in ids)
        col_left[depth] = cursor
        cursor += col_w / 2
        col_x[depth] = cursor
        cursor += col_w / 2
        col_right[depth] = cursor
        cursor += COL_GAP
    right = cursor - COL_GAP

    # Modalities are spread over the full span rather than packed into a short
    # stack, which keeps the edges leaving them short and roughly parallel.
    pos = {}
    step = (span - MOD_H) / (len(MODALITY_ORDER) - 1)
    for i, modality in enumerate(MODALITY_ORDER):
        pos[f'mod:{modality}'] = TOP - MOD_H / 2 - i * step

    # Everything else sits at the mean height of the nodes it connects to, which
    # settles the column in one step. On the first sweep only the inputs are
    # placed; later sweeps see the consumers too.
    for _ in range(SWEEPS):
        for depth, ids in layers.items():
            if depth == 0:
                continue
            desired = {}
            for node_id in ids:
                neighbours = [
                    pos[neighbour]
                    for neighbour in list(pred.get(node_id, ())) + list(succ.get(node_id, ()))
                    if neighbour in pos
                ]
                desired[node_id] = (
                    sum(neighbours) / len(neighbours) if neighbours else TOP - span / 2
                )
            pos.update(place_column(ids, desired, heights, bottom))

    floor = min(pos[node_id] - heights[node_id] / 2 for node_id in nodes)

    backbone = [
        node_id
        for node_id, node in nodes.items()
        if node['depth'] == 1 and f'mod:{UNIVERSAL}' not in node['inputs']
    ]

    # Fan each node's hops across its own left and right sides so they do not all
    # leave from or land on a single point. Landings are ordered by the height
    # each hop leaves from, so arriving lines keep their relative order instead
    # of crossing at the box.
    outgoing = {node_id: [] for node_id in placed}
    incoming = {node_id: [] for node_id in placed}
    for src, dst in segments:
        outgoing[src].append(dst)
        incoming[dst].append(src)
    for dst in backbone:
        outgoing[f'mod:{UNIVERSAL}'].append(dst)
        incoming[dst].append(f'mod:{UNIVERSAL}')

    anchors = {}
    for src, targets in outgoing.items():
        targets.sort(key=lambda dst: pos[dst], reverse=True)
        for i, dst in enumerate(targets):
            offset = 0.0 if len(targets) == 1 else (i / (len(targets) - 1) - 0.5)
            anchors[(src, dst)] = pos[src] - offset * heights[src] * 0.62

    landings = {}
    for dst, sources in incoming.items():
        sources.sort(key=lambda src: anchors[(src, dst)], reverse=True)
        for i, src in enumerate(sources):
            offset = 0.0 if len(sources) == 1 else (i / (len(sources) - 1) - 0.5)
            landings[(src, dst)] = pos[dst] - offset * heights[dst] * 0.58

    def edge_polyline(chain):
        """Waypoints for one edge, from the source box edge to the target box edge.

        Every column is crossed horizontally, at the height of the box the edge
        leaves, the waypoint it was given, or the box it lands on; all the
        vertical travel happens in the gutters between columns, where there is
        nothing to hit. That is what keeps an edge out of the boxes even when it
        leaves a narrow box standing beside a much wider one, which a straight
        run from box edge to box edge would cut straight through.
        """
        src, dst = chain[0], chain[-1]
        src_depth, dst_depth = nodes[src]['depth'], nodes[dst]['depth']

        start_y = anchors[(src, chain[1])]
        points = [(col_x[src_depth] + widths[src] / 2, start_y), (col_right[src_depth], start_y)]

        for via in chain[1:-1]:
            depth = placed[via]['depth']
            points += [(col_left[depth], pos[via]), (col_right[depth], pos[via])]

        end_y = landings[(chain[-2], dst)]
        points += [(col_left[dst_depth], end_y), (col_x[dst_depth] - widths[dst] / 2, end_y)]
        return points

    routes = OrderedDict(
        ((chain[0], chain[-1]), edge_polyline(chain))
        for chain in list(chains.values()) + [[f'mod:{UNIVERSAL}', dst] for dst in backbone]
    )
    check_no_box_intrusions(
        routes,
        {
            node_id: (
                col_x[node['depth']] - widths[node_id] / 2,
                col_x[node['depth']] + widths[node_id] / 2,
                pos[node_id] - heights[node_id] / 2,
                pos[node_id] + heights[node_id] / 2,
            )
            for node_id, node in nodes.items()
        },
    )

    x0, x1 = -2.0, right + 2.0
    y0, y1 = floor - 16, TOP + 8
    fig, ax = plt.subplots(figsize=((x1 - x0) / SCALE, (y1 - y0) / SCALE))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.axis('off')

    for depth, x in col_x.items():
        ax.text(
            x,
            TOP + 5,
            COLUMN_HEADERS[depth],
            ha='center',
            va='center',
            fontsize=FONT_HEADER,
            fontweight='bold',
            color=INK_MUTED,
        )

    # Backbone edges first, so the real dependencies draw over them.
    for dst in backbone:
        points = edge_polyline([f'mod:{UNIVERSAL}', dst])
        draw_spline(ax, points, color=FAINT, linewidth=0.7, head=(3.5, 2.2), zorder=1)

    for chain in chains.values():
        draw_spline(ax, edge_polyline(chain))

    for node_id, node in nodes.items():
        role = 'modality' if node['kind'] == 'mod' else 'output'
        fontsize = FONT_MOD if node['kind'] == 'mod' else FONT_SET
        draw_box(
            ax,
            col_x[node['depth']],
            pos[node_id],
            widths[node_id],
            heights[node_id],
            node['headline'],
            node['detail'],
            role,
            fontsize=fontsize,
        )

    # Legend: two node roles and two edge weights, in two columns so it stays
    # inside the figure width instead of stretching the canvas.
    row_h = 3.6
    entries = [
        ('box', 'modality', 'Source modality'),
        ('box', 'output', 'Scalar map, or set sharing one dependency signature'),
        ('edge', 'direct', 'Direct input'),
        ('edge', 'faint', 'Anatomical reference via sMRIPrep (cross-session)'),
    ]
    for i, (kind, role, label) in enumerate(entries):
        lx = x0 + 2 + (i % 2) * (right / 2)
        ly = floor - 8 - (i // 2) * row_h
        if kind == 'box':
            ax.add_patch(
                FancyBboxPatch(
                    (lx, ly - 0.9),
                    3.0,
                    1.8,
                    boxstyle='round,pad=0,rounding_size=0.3',
                    facecolor=tint(PALETTE[role]),
                    edgecolor=PALETTE[role],
                    linewidth=1.5,
                    zorder=3,
                )
            )
        elif role == 'direct':
            draw_edge(ax, (lx, ly), (lx + 3.0, ly))
        else:
            draw_edge(ax, (lx, ly), (lx + 3.0, ly), color=FAINT, linewidth=0.7, head=(3.5, 2.2))
        ax.text(lx + 4.0, ly, label, ha='left', va='center', fontsize=FONT_LEGEND, color=INK_MUTED)

    save(fig, 'workflow_modality_layers')
    plt.close(fig)
