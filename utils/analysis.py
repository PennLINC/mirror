"""Utilities for analysis scripts.

Provides ``convert_to_multindex`` for reshaping flat DataFrames into
MultiIndex form, ``matrix`` for visualizing nullity/missingness, and
``get_qc_passing_subjects`` for applying modality-level manual QC to scalars.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def dice(array_a: np.ndarray, array_b: np.ndarray) -> float:
    """Return the Sørensen-Dice overlap of two arrays after binarization."""

    mask_a = np.asarray(array_a).astype(bool)
    mask_b = np.asarray(array_b).astype(bool)
    denominator = int(mask_a.sum() + mask_b.sum())
    if denominator == 0:
        return 0.0
    return float(2 * np.count_nonzero(mask_a & mask_b) / denominator)


def get_qc_passing_subjects(
    qc_df: pd.DataFrame,
    modalities: list[str],
    session: str,
) -> list[str]:
    """Find subjects for whom every listed modality passed manual QC in a session.

    A scalar is only usable if all of the modalities that went into it passed, so
    callers should pass the modality list for the scalar of interest (see
    the ``qc_modalities`` field in ``configuration/metrics.yml``).

    Parameters
    ----------
    qc_df : pandas.DataFrame
        Manual QC ratings indexed by participant ID, with 'Session <NN>--<modality>'
        columns (see ``data/qc/manual_qc_modality.tsv``).
    modalities : list of str
        Modalities used to create the scalar.
    session : str
        Session label without the 'ses-' prefix (e.g., '01').

    Returns
    -------
    list of str
        Participant IDs that passed QC for every one of ``modalities``.
    """
    columns = [f'Session {session}--{modality}' for modality in modalities]
    missing = [column for column in columns if column not in qc_df.columns]
    if missing:
        raise KeyError(f'No QC columns found: {missing}')

    # Anything other than a pass (1) is treated as a failure, including n/a
    return qc_df.index[(qc_df[columns] == 1).all(axis=1)].tolist()


def convert_to_multindex(
    df: pd.DataFrame,
    separator: str = '--',
    level_names: list[str] | None = None,
) -> pd.DataFrame:
    """Convert DataFrame columns from 'parent--child' pattern to MultiIndex.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns in 'parent--child' format
    separator : str, default '--'
        Separator used between parent and child names
    level_names : list, optional
        Names for the MultiIndex levels (e.g., ['Parent', 'Child'])

    Returns
    -------
    pandas.DataFrame
        DataFrame with MultiIndex columns

    Examples
    --------
    >>> df = pd.DataFrame({'Group A--Metric 1': [1, 2], 'Group A--Metric 2': [3, 4]})
    >>> df_multindex = convert_to_multindex(df)
    >>> print(df_multindex.columns)
    MultiIndex([('Group A', 'Metric 1'),
                ('Group A', 'Metric 2')],
               names=['Level_0', 'Level_1'])
    """
    if not isinstance(df.columns, pd.Index):
        raise ValueError('Input must be a pandas DataFrame')

    # Check if columns already contain the separator
    if not any(separator in str(col) for col in df.columns):
        raise ValueError(f"No columns found with separator '{separator}'")

    # Convert columns to MultiIndex
    new_columns = []
    for col in df.columns:
        if separator in str(col):
            parts = str(col).split(separator, 1)  # Split only on first occurrence
            if len(parts) == 2:
                new_columns.append((parts[0].strip(), parts[1].strip()))
            else:
                # If splitting doesn't work as expected, keep original column
                new_columns.append((str(col), ''))
        else:
            # Column doesn't have separator, treat as single level
            new_columns.append((str(col), ''))

    # Create MultiIndex
    multindex = pd.MultiIndex.from_tuples(new_columns, names=level_names)

    # Create new DataFrame with MultiIndex columns
    df_new = df.copy()
    df_new.columns = multindex

    return df_new


def format_completeness_label(value: float) -> str:
    """Format sparkline endpoint labels without decimal noise for integer counts."""

    return str(int(round(value))) if np.isclose(value, round(value)) else f'{value:g}'


def matrix(
    df: pd.DataFrame,
    nullity_filter: str | None = None,
    n: int = 0,
    p: int = 0,
    sort: str | None = None,
    figsize: tuple[int, int] = (25, 15),
    width_ratios: tuple[int, int] = (15, 1),
    color: tuple[float, float, float] = (0.25, 0.25, 0.25),
    fontsize: int = 16,
    labels: bool | None = None,
    label_rotation: int = 45,
    sparkline: bool = True,
    ax: object | None = None,
    palette: list | None = None,
    excluded: pd.DataFrame | None = None,
    excluded_color: tuple[float, float, float] = (0.75, 0.75, 0.75),
    sparkline_values: pd.Series | np.ndarray | list | None = None,
) -> object:
    """A matrix visualization of the nullity of the given DataFrame.

    Modified from https://github.com/ResidentMario/missingno/blob/master/missingno/missingno.py.

    Changes:
    - Added palette parameter
    - Support for MultiIndex columns
    - Dropped freq argument
    - Added horizontal grid lines
    - Corrected offset of sparkline points, line, and labels
    - Increased font size.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame being mapped.
    nullity_filter : str, optional
        The filter to apply to the heatmap. Should be one of "top", "bottom", or None (default).
    n : int, optional
        The max number of columns to include in the filtered DataFrame.
    p : int, optional
        The max percentage fill of the columns in the filtered DataFrame.
    sort : str, optional
        The row sort order to apply. Can be "ascending", "descending", or None.
    figsize : tuple, optional
        The size of the figure to display.
    width_ratios : tuple, optional
        The ratio of the width of the matrix to the width of the sparkline. Defaults to `(15, 1)`.
        Does nothing if `sparkline=False`.
    color : tuple, optional
        The color of the filled columns. Default is `(0.25, 0.25, 0.25)`.
    fontsize : int, optional
        The figure's font size. Default to 16.
    labels : bool, optional
        Whether or not to display the column names. Defaults to the underlying data labels when there are
        50 columns or less, and no labels when there are more than 50 columns.
    label_rotation : int, optional
        What angle to rotate the text labels to. Defaults to 45 degrees.
    sparkline : bool, optional
        Whether or not to display the sparkline. Defaults to True.
    ax : matplotlib.axes.Axes, optional
        The plot axis. Defaults to None.
    palette : list, optional
        The palette to use for the heatmap. Defaults to None.
    excluded : pandas.DataFrame, optional
        Boolean DataFrame aligned to `df` marking cells that are present but should be grayed
        out (e.g., images that failed manual QC). Missing cells are left white regardless.
    excluded_color : tuple, optional
        The color of the excluded cells. Default is `(0.75, 0.75, 0.75)`.
    sparkline_values : pandas.Series or array-like, optional
        Row-wise values to plot in the sparkline. Defaults to the row-wise sum
        of `df`, i.e. the number of present cells.

    Returns
    -------
    matplotlib.axes.Axes
        The plot axis.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.patches import Polygon
    from missingno.utils import nullity_filter as _nullity_filter, nullity_sort

    df = _nullity_filter(df, filter=nullity_filter, n=n, p=p)
    df = nullity_sort(df, sort=sort, axis='columns')

    height = df.shape[0]
    width = df.shape[1]

    # z is the color-mask array, g is a NxNx3 matrix. Apply the z color-mask to set the RGB of each pixel.
    z = df.values
    g = np.zeros((height, width, 3), dtype=np.float32)

    g[z < 0.5] = [1, 1, 1]
    g[z >= 0.5] = color
    # TDS
    if palette is not None:
        for i_col in range(width):
            g[z[:, i_col] >= 0.5, i_col] = palette[i_col]

    # TDS: Gray out cells that were acquired but excluded during manual QC.
    if excluded is not None:
        excluded_mask = (
            excluded.reindex(index=df.index, columns=df.columns).fillna(False).values.astype(bool)
        )
        g[excluded_mask & (z >= 0.5)] = excluded_color

    # Set up the matplotlib grid layout. A unary subplot if no sparkline, a left-right splot if yes sparkline.
    if ax is None:
        plt.figure(figsize=figsize)

        # Check if we have MultiIndex columns to determine layout
        has_multindex = isinstance(df.columns, pd.MultiIndex)
        parent_levels = df.columns.nlevels - 1 if has_multindex else 0

        if sparkline:
            # Create grid with space for parent levels, main plot, and sparkline
            total_rows = parent_levels + 2
            gs = gridspec.GridSpec(
                total_rows,
                2,
                width_ratios=width_ratios,
                height_ratios=[0.1] * (parent_levels + 1) + [1],
            )
            gs.update(wspace=0.08, hspace=0.1)

            # Create parent level subplots
            parent_axes = []
            for level_idx in range(parent_levels):
                parent_ax = plt.subplot(gs[level_idx, 0])
                parent_axes.append(parent_ax)

            # Main plot and sparkline
            ax0 = plt.subplot(gs[parent_levels + 1, 0])
            ax1 = plt.subplot(gs[parent_levels + 1, 1])
        else:
            # Create grid with space for parent levels and main plot
            total_rows = parent_levels + 1
            gs = gridspec.GridSpec(total_rows, 1, height_ratios=[0.1] * parent_levels + [1])
            gs.update(hspace=0.1)

            # Create parent level subplots
            parent_axes = []
            for level_idx in range(parent_levels):
                parent_ax = plt.subplot(gs[level_idx, 0])
                parent_axes.append(parent_ax)

            # Main plot
            ax0 = plt.subplot(gs[parent_levels, 0])
    else:
        if sparkline is not False:
            warnings.warn(
                'Plotting a sparkline on an existing axis is not currently supported. '
                'To remove this warning, set sparkline=False.'
            )
            sparkline = False
        ax0 = ax
        parent_axes = []  # No parent axes when using existing axis

    # Create the nullity plot.
    ax0.imshow(g, interpolation='none')

    # Overlay a diagonal triangle on cells where the diagonal value is neither 0 nor 1
    values_array = df.values
    for i_row in range(height):
        for j_col in range(width):
            v = values_array[i_row, j_col]
            if not np.isclose(v, 0.0) and not np.isclose(v, 1.0):
                # Coordinates for the lower-right triangle of cell (i_row, i_col)
                x0, x1 = j_col - 0.5, j_col + 0.5
                y0, y1 = i_row - 0.5, i_row + 0.5
                tri = Polygon(
                    [(x1, y0), (x1, y1), (x0, y1)],
                    closed=True,
                    fill=True,
                    facecolor='white',
                    edgecolor='white',
                    linewidth=1.5,
                    zorder=3,
                )
                ax0.add_patch(tri)

    # Remove extraneous default visual elements.
    ax0.set_aspect('auto')
    ax0.grid(visible=False)
    ax0.xaxis.tick_top()
    ax0.xaxis.set_ticks_position('none')
    ax0.yaxis.set_ticks_position('none')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)

    # Set up and rotate the column ticks. The labels argument is set to None by default. If the user specifies it in
    # the argument, respect that specification. Otherwise display for <= 50 columns and do not display for > 50.
    if labels or (labels is None and len(df.columns) <= 50):
        ha = 'left'
        ax0.set_xticks(list(range(0, width)))

        # Support MultiIndex columns with centered parent level names
        if isinstance(df.columns, pd.MultiIndex) and parent_axes:
            # Get child column labels (bottom level)
            child_labels = [str(col) for col in df.columns.get_level_values(-1)]
            ax0.set_xticklabels(child_labels, rotation=label_rotation, ha=ha, fontsize=fontsize)

            # Add parent level labels to dedicated subplots
            for level_idx, parent_ax in enumerate(parent_axes):
                level_values = df.columns.get_level_values(level_idx)

                # Group consecutive identical parent values
                parent_groups = []
                current_group = []
                current_parent = None

                for i, parent_val in enumerate(level_values):
                    if parent_val != current_parent:
                        if current_group:
                            parent_groups.append((current_parent, current_group))
                        current_group = [i]
                        current_parent = parent_val
                    else:
                        current_group.append(i)

                if current_group:
                    parent_groups.append((current_parent, current_group))

                # Set up parent axis
                parent_ax.set_xlim(-0.5, width - 0.5)
                parent_ax.set_ylim(0, 1)
                parent_ax.set_xticks([])
                parent_ax.set_yticks([])

                # Remove all spines
                for spine in parent_ax.spines.values():
                    spine.set_visible(False)

                # Add parent labels and lines
                for parent_val, child_indices in parent_groups:
                    start_idx = min(child_indices)
                    end_idx = max(child_indices)
                    center_pos = (start_idx + end_idx) / 2

                    # Add horizontal line
                    parent_ax.plot(
                        [start_idx, end_idx],
                        [0.3, 0.3],
                        'k-',
                        linewidth=3,
                        alpha=1,
                    )

                    # Add parent label
                    parent_ax.text(
                        center_pos,
                        0.7,
                        str(parent_val),
                        ha='center',
                        va='center',
                        fontsize=fontsize * 1.2,
                        weight='bold',
                    )
        elif isinstance(df.columns, pd.MultiIndex):
            # Fallback for when using existing axis (no parent_axes available)
            child_labels = [str(col) for col in df.columns.get_level_values(-1)]
            ax0.set_xticklabels(child_labels, rotation=label_rotation, ha=ha, fontsize=fontsize)
        else:
            # Regular single-level columns
            ax0.set_xticklabels(list(df.columns), rotation=label_rotation, ha=ha, fontsize=fontsize)
    else:
        ax0.set_xticks([])

    # Modification: Drop freq argument.
    # Modification: use the index values instead of first and last row numbers.
    ax0.set_yticks(range(0, df.shape[0]))
    ax0.set_yticklabels(df.index.tolist(), fontsize=int(fontsize / 16 * 20), rotation=0)

    # Create the inter-column vertical grid.
    in_between_point = [x + 0.5 for x in range(0, width - 1)]
    for in_between_point in in_between_point:
        ax0.axvline(in_between_point, linestyle='-', color='white')

    # Create the inter-row horizontal grid (TDS)
    in_between_point = [y + 0.5 for y in range(0, height - 1)]
    for in_between_point in in_between_point:
        ax0.axhline(in_between_point, linestyle='-', color='white')

    if sparkline:
        # Calculate row-wise completeness for the sparkline.
        if sparkline_values is None:
            completeness_srs = df.values.sum(axis=1)
        elif isinstance(sparkline_values, pd.Series):
            completeness_srs = sparkline_values.reindex(df.index).to_numpy(dtype=float)
        else:
            completeness_srs = np.asarray(sparkline_values, dtype=float)
            if completeness_srs.shape[0] != height:
                raise ValueError(
                    'sparkline_values must have one value per row after filtering/sorting.'
                )
        x_domain = list(range(0, height))
        y_range = list(reversed(completeness_srs))
        min_completeness = min(y_range)
        max_completeness = max(y_range)
        min_completeness_index = y_range.index(min_completeness)
        max_completeness_index = y_range.index(max_completeness)

        # Set up the sparkline, remove the border element.
        ax1.grid(visible=False)
        ax1.set_aspect('auto')
        # GH 25
        if int(mpl.__version__[0]) <= 1:
            ax1.set_axis_bgcolor((1, 1, 1))
        else:
            ax1.set_facecolor((1, 1, 1))
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.set_ymargin(0)

        # Plot sparkline---plot is sideways so the x and y axis are reversed.
        # Modification: offset row values by 0.5 to center the points on the rows.
        x_domain = [i + 0.5 for i in x_domain]
        ax1.plot(y_range, x_domain, color=color)

        if labels:
            # Figure out what case to display the label in: mixed, upper, lower.
            label = 'Data Completeness'
            if str(df.columns[0]).islower():
                label = label.lower()
            if str(df.columns[0]).isupper():
                label = label.upper()

            # Set up and rotate the sparkline label.
            ha = 'left'
            ax1.set_xticks([min_completeness + (max_completeness - min_completeness) / 2])
            ax1.set_xticklabels([label], rotation=label_rotation, ha=ha, fontsize=fontsize)
            ax1.xaxis.tick_top()
            ax1.set_yticks([])
        else:
            ax1.set_xticks([])
            ax1.set_yticks([])

        # Add maximum and minimum labels, circles.
        # Modification: offset row values by 0.5 to center the points on the rows.
        ax1.annotate(
            format_completeness_label(max_completeness),
            xy=(max_completeness, max_completeness_index + 0.5),
            xytext=(max_completeness + 2, max_completeness_index + 0.5),
            fontsize=int(fontsize / 16 * 14),
            va='center',
            ha='left',
        )
        ax1.annotate(
            format_completeness_label(min_completeness),
            xy=(min_completeness, min_completeness_index + 0.5),
            xytext=(min_completeness - 2, min_completeness_index + 0.5),
            fontsize=int(fontsize / 16 * 14),
            va='center',
            ha='right',
        )

        ax1.set_xlim(
            [min_completeness - 2, max_completeness + 2]
        )  # Otherwise the circles are cut off.
        ax1.set_ylim(0, height)
        # Modification: offset row values by 0.5 to center the points on the rows.
        ax1.plot(
            [min_completeness], [min_completeness_index + 0.5], '.', color=color, markersize=10.0
        )
        ax1.plot(
            [max_completeness], [max_completeness_index + 0.5], '.', color=color, markersize=10.0
        )

        # Remove tick mark (only works after plotting).
        ax1.xaxis.set_ticks_position('none')

    return ax0
