"""Tests for analysis utilities -- Tier 1 (unit).

Covers shared analysis utilities used by active figure code.
"""

import numpy as np
import pandas as pd
import pytest

from utils.analysis import convert_to_multindex, dice


# ===================================================================
# dice
# ===================================================================


class TestDice:
    """Unit tests for the Dice coefficient calculation."""

    def test_identical_arrays(self):
        a = np.array([1, 1, 0, 0, 1])
        assert dice(a, a) == 1.0

    def test_disjoint_arrays(self):
        a = np.array([1, 1, 0, 0])
        b = np.array([0, 0, 1, 1])
        assert dice(a, b) == 0.0

    def test_both_empty(self):
        a = np.zeros(5)
        b = np.zeros(5)
        assert dice(a, b) == 0

    def test_partial_overlap(self):
        """Two arrays sharing exactly one of three nonzero elements each."""
        a = np.array([1, 1, 1, 0, 0])
        b = np.array([0, 0, 1, 1, 1])
        # intersection=1, |A|=3, |B|=3 => DC = 2*1 / (3+3) = 1/3
        np.testing.assert_allclose(dice(a, b), 1 / 3)

    def test_nonbinary_input(self):
        """Non-binary values should be binarized (0 => False, else => True)."""
        a = np.array([5, 0, 3])
        b = np.array([0, 0, 7])
        # After binarization: a=[T,F,T], b=[F,F,T]
        # intersection=1, |A|=2, |B|=1 => DC = 2*1/3
        np.testing.assert_allclose(dice(a, b), 2 / 3)

    def test_float_input(self):
        a = np.array([0.5, 0.0, 1.0])
        b = np.array([0.5, 0.5, 0.0])
        # Binarized: a=[T,F,T], b=[T,T,F]
        # intersection=1, |A|=2, |B|=2 => DC = 2/4 = 0.5
        assert dice(a, b) == 0.5

    def test_2d_arrays(self):
        """Dice should work on multi-dimensional arrays."""
        a = np.array([[1, 0], [1, 1]])
        b = np.array([[1, 1], [0, 1]])
        # After flatten/binarize: a=[T,F,T,T], b=[T,T,F,T]
        # intersection=2, |A|=3, |B|=3 => DC = 4/6
        np.testing.assert_allclose(dice(a, b), 4 / 6)


# ===================================================================
# convert_to_multindex
# ===================================================================


class TestConvertToMultindex:
    """Unit tests for the DataFrame column MultiIndex converter."""

    def test_basic_conversion(self):
        df = pd.DataFrame(
            {
                'Session 01--MPRAGE': [1, 2],
                'Session 01--FLAIR': [3, 4],
                'Session 02--MPRAGE': [5, 6],
            }
        )
        result = convert_to_multindex(df)
        assert isinstance(result.columns, pd.MultiIndex)
        assert result.columns.nlevels == 2
        expected = [
            ('Session 01', 'MPRAGE'),
            ('Session 01', 'FLAIR'),
            ('Session 02', 'MPRAGE'),
        ]
        assert list(result.columns) == expected

    def test_data_preserved(self):
        df = pd.DataFrame(
            {
                'A--X': [10, 20],
                'A--Y': [30, 40],
            }
        )
        result = convert_to_multindex(df)
        np.testing.assert_array_equal(result.values, df.values)

    def test_no_separator_raises(self):
        df = pd.DataFrame({'col_a': [1], 'col_b': [2]})
        with pytest.raises(ValueError, match='No columns found with separator'):
            convert_to_multindex(df)

    def test_custom_separator(self):
        df = pd.DataFrame({'parent::child': [1]})
        result = convert_to_multindex(df, separator='::')
        assert list(result.columns) == [('parent', 'child')]

    def test_custom_level_names(self):
        df = pd.DataFrame({'A--B': [1]})
        result = convert_to_multindex(df, level_names=['Parent', 'Child'])
        assert result.columns.names == ['Parent', 'Child']

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace around the separator should be stripped."""
        df = pd.DataFrame({'Group A -- Metric 1': [1]})
        result = convert_to_multindex(df)
        assert result.columns[0] == ('Group A', 'Metric 1')
