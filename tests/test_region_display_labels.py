"""Tests for publication-facing parcel and bundle labels."""

import pytest

from utils.regions import (
    dkt_parcel_display_label,
    regional_feature_display_label,
    wm_bundle_display_label,
)


@pytest.mark.parametrize(
    ('track_id', 'expected'),
    [
        ('Association_ArcuateFasciculusL', 'Arcuate Fasciculus (L)'),
        ('Association_ArcuateFasciculusR', 'Arcuate Fasciculus (R)'),
        ('Commissure_CorpusCallosum_Body', 'Corpus Callosum Body'),
        (
            'ProjectionBrainstem_DentatorubrothalamicTract-lr',
            'Dentato-Rubro-Thalamic Tract (L→R)',
        ),
    ],
)
def test_wm_bundle_display_label(track_id, expected):
    assert wm_bundle_display_label(track_id) == expected


@pytest.mark.parametrize(
    ('parcel_id', 'expected'),
    [
        ('rh_pericalcarine', 'Pericalcarine (R)'),
        ('lh_caudal anterior cingulate', 'Caudal Anterior Cingulate (L)'),
    ],
)
def test_dkt_parcel_display_label(parcel_id, expected):
    assert dkt_parcel_display_label(parcel_id) == expected


def test_regional_feature_display_label_routes_by_tissue():
    assert regional_feature_display_label(
        'Association_ArcuateFasciculusL', 'wm'
    ) == 'Arcuate Fasciculus (L)'
    assert regional_feature_display_label(
        'rh_pericalcarine', 'gm'
    ) == 'Pericalcarine (R)'


def test_regional_feature_display_label_rejects_unknown_tissue():
    with pytest.raises(ValueError, match='Unsupported regional tissue'):
        regional_feature_display_label('feature', 'csf')
