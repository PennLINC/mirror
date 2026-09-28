"""Tests for g-ratio scaling-factor processing -- Tier 1 (unit)."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np

# The module imports heavy deps at the top level and calls load_config() at
# module scope.  Mock the heavy deps and patch load_config before importing.
sys.modules.setdefault('ants', MagicMock())
sys.modules.setdefault('nilearn', MagicMock())
sys.modules.setdefault('nilearn.masking', MagicMock())
sys.modules.setdefault('nilearn.plotting', MagicMock())

with patch(
    'utils.processing.load_config',
    return_value={
        'code_dir': '/fake/code',
        'bids_dir': '/fake/bids',
        'work_dir': '/fake/work',
        'derivatives': {},
    },
):
    # Force a clean import that picks up the patched load_config
    sys.modules.pop('process_g_ratio_scaling_factors', None)
    from process_g_ratio_scaling_factors import compute_scaling_factor


class TestComputeScalingFactor:
    """Unit tests for the g-ratio scaling factor computation."""

    def test_round_trip_default_g(self):
        """Plug the scaling factor back into the g-ratio formula and verify g == 0.7."""
        ICVF, MVF, ISOVF = 0.6, 0.15, 0.1
        sf = compute_scaling_factor(ICVF, MVF, ISOVF, g=0.7)

        # Reconstruct g-ratio from the scaling factor
        MVFs = MVF * sf
        FVF = (1 - MVFs) * (1 - ISOVF) * ICVF
        g_recon = np.sqrt(FVF / (FVF + MVFs))

        np.testing.assert_allclose(g_recon, 0.7, atol=1e-10)

    def test_round_trip_custom_g(self):
        """Same round-trip with a non-default target g-ratio."""
        ICVF, MVF, ISOVF = 0.5, 0.2, 0.05
        for g_target in [0.5, 0.6, 0.8, 0.9]:
            sf = compute_scaling_factor(ICVF, MVF, ISOVF, g=g_target)
            MVFs = MVF * sf
            FVF = (1 - MVFs) * (1 - ISOVF) * ICVF
            g_recon = np.sqrt(FVF / (FVF + MVFs))
            np.testing.assert_allclose(g_recon, g_target, atol=1e-10)

    def test_array_inputs(self):
        """Function should broadcast over array inputs."""
        ICVF = np.array([0.5, 0.6, 0.7])
        MVF = np.array([0.10, 0.15, 0.20])
        ISOVF = np.array([0.05, 0.10, 0.15])
        sf = compute_scaling_factor(ICVF, MVF, ISOVF, g=0.7)

        assert sf.shape == (3,)

        # Round-trip check for each element
        MVFs = MVF * sf
        FVF = (1 - MVFs) * (1 - ISOVF) * ICVF
        g_recon = np.sqrt(FVF / (FVF + MVFs))
        np.testing.assert_allclose(g_recon, 0.7, atol=1e-10)

    def test_mvf_zero_division(self):
        """MVF == 0 should produce inf (division by zero in denominator)."""
        sf = compute_scaling_factor(ICVF=0.6, MVF=0.0, ISOVF=0.1, g=0.7)
        assert np.isinf(sf)

    def test_higher_g_gives_lower_scaling_factor(self):
        """Higher target g-ratio should require a smaller scaling factor.

        A higher g means thinner myelin relative to fiber diameter.
        With fixed ICVF/ISOVF, less scaled MVF is needed, so the scaling
        factor should be smaller.
        """
        ICVF, MVF, ISOVF = 0.6, 0.15, 0.1
        sf_low = compute_scaling_factor(ICVF, MVF, ISOVF, g=0.6)
        sf_high = compute_scaling_factor(ICVF, MVF, ISOVF, g=0.8)
        assert sf_high < sf_low

    def test_scalar_output_type(self):
        """Scalar inputs should return a scalar (float)."""
        sf = compute_scaling_factor(ICVF=0.6, MVF=0.15, ISOVF=0.1, g=0.7)
        assert isinstance(sf, (float, np.floating))
