"""Publication-facing labels for regional parcel and bundle identifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedTrack:
    """Normalized components of a DSI Studio AutoTrack identifier."""

    category: str
    bundle: str
    side: str | None
    crossed_direction: str | None


def _humanize_bundle(value: str) -> str:
    """Convert an AutoTrack bundle token to a readable anatomical label."""

    words = value.replace('_', ' ')
    words = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', words)
    words = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', words)
    words = re.sub(r'(?<=[A-Za-z])(?=[0-9])', ' ', words)
    words = re.sub(r'\s+', ' ', words).strip().lower()
    replacements = {
        'fronto occipital': 'fronto-occipital',
        'parahippocampal parietal': 'parahippocampal–parietal',
        'dentatorubrothalamic': 'dentato-rubro-thalamic',
        'non decussating': 'non-decussating',
    }
    for source, target in replacements.items():
        words = words.replace(source, target)
    return words[:1].upper() + words[1:]


def parse_track_id(track_id: str) -> ParsedTrack:
    """Parse a categorized DSI Studio AutoTrack identifier."""

    if '_' not in track_id:
        raise ValueError(f'Unexpected AutoTrack identifier without category: {track_id}')
    category, raw_bundle = track_id.split('_', 1)

    crossed_direction = None
    crossed_match = re.search(r'-(lr|rl)$', raw_bundle, flags=re.IGNORECASE)
    if crossed_match:
        crossed_direction = crossed_match.group(1).lower()
        raw_bundle = raw_bundle[: crossed_match.start()]

    side_match = re.search(r'(?<=[a-z0-9])([LR])(?=_|$)', raw_bundle)
    side = side_match.group(1) if side_match else None
    if side_match:
        raw_bundle = raw_bundle[: side_match.start()] + raw_bundle[side_match.end() :]

    return ParsedTrack(
        category=category,
        bundle=_humanize_bundle(raw_bundle),
        side=side,
        crossed_direction=crossed_direction,
    )


def wm_bundle_display_label(track_id: object) -> str:
    """Convert an AutoTrack identifier to an anatomical name and laterality."""

    parsed = parse_track_id(str(track_id))
    label = parsed.bundle.title()
    if parsed.crossed_direction == 'lr':
        return f'{label} (L→R)'
    if parsed.crossed_direction == 'rl':
        return f'{label} (R→L)'
    if parsed.side in {'L', 'R'}:
        return f'{label} ({parsed.side})'
    return label


def dkt_parcel_display_label(parcel_id: object) -> str:
    """Convert a DKT ``lh_/rh_`` identifier to a readable cortical label."""

    value = str(parcel_id).strip()
    match = re.match(r'^(lh|rh|l|r)[_\s-]+(.+)$', value, flags=re.IGNORECASE)
    if match is None:
        return value.replace('_', ' ').title()

    hemisphere = 'L' if match.group(1).lower() in {'lh', 'l'} else 'R'
    region = re.sub(r'[_\s-]+', ' ', match.group(2)).strip().title()
    return f'{region} ({hemisphere})'


def regional_feature_display_label(feature: object, tissue: str) -> str:
    """Return the appropriate publication label for a regional feature."""

    if tissue == 'wm':
        return wm_bundle_display_label(feature)
    if tissue == 'gm':
        return dkt_parcel_display_label(feature)
    raise ValueError(f'Unsupported regional tissue: {tissue}')
