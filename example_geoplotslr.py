import missingno as msno
import numpy as np
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.pyplot as plt
import pandas as pd


def nullity_sort(df, sort=None):
    """
    Sorts a DataFrame according to its nullity, in either ascending or descending order.
    :param df: The DataFrame object being sorted.
    :param sort: The sorting method: either "ascending", "descending", or None (default).
    :return: The nullity-sorted DataFrame.
    """
    if sort == 'ascending':
        return df.iloc[np.argsort(df.count(axis='columns').values), :]
    elif sort == 'descending':
        return df.iloc[np.flipud(np.argsort(df.count(axis='columns').values)), :]
    else:
        return df


def nullity_filter(df, filter=None, p=0, n=0):
    """
    Filters a DataFrame according to its nullity, using some combination of 'top' and 'bottom' numerical and
    percentage values. Percentages and numerical thresholds can be specified simultaneously: for example,
    to get a DataFrame with columns of at least 75% completeness but with no more than 5 columns, use
    `nullity_filter(df, filter='top', p=.75, n=5)`.
    :param df: The DataFrame whose columns are being filtered.
    :param filter: The orientation of the filter being applied to the DataFrame. One of, "top", "bottom",
    or None (default). The filter will simply return the DataFrame if you leave the filter argument unspecified or
    as None.
    :param p: A completeness ratio cut-off. If non-zero the filter will limit the DataFrame to columns with at least p
    completeness. Input should be in the range [0, 1].
    :param n: A numerical cut-off. If non-zero no more than this number of columns will be returned.
    :return: The nullity-filtered `DataFrame`.
    """
    if filter == 'top':
        if p:
            df = df.iloc[:, [c >= p for c in df.count(axis='rows').values / len(df)]]
        if n:
            df = df.iloc[:, np.sort(np.argsort(df.count(axis='rows').values)[-n:])]
    elif filter == 'bottom':
        if p:
            df = df.iloc[:, [c <= p for c in df.count(axis='rows').values / len(df)]]
        if n:
            df = df.iloc[:, np.sort(np.argsort(df.count(axis='rows').values)[:n])]
    return df


def matrix_new(df,
               filter=None, n=0, p=0, sort=None, reindex=None, drop=None,
               figsize=(25, 10), width_ratios=(15, 1), color=(0.25, 0.25, 0.25),
               fontsize=16, labels=None, sparkline=True, inline=True,
               freq=None):
    """
    A matrix visualization of the nullity of the given DataFrame.

    For optimal performance, please stay within 250 rows and 50 columns.

    :param df: The `DataFrame` being mapped.
    :param filter: The filter to apply to the heatmap. Should be one of "top", "bottom", or None (default).
    :param n: The max number of columns to include in the filtered DataFrame.
    :param p: The max percentage fill of the columns in the filtered DataFrame.
    :param sort: The sort to apply to the heatmap. Should be one of "ascending", "descending", or None (default).
    :param figsize: The size of the figure to display.
    :param fontsize: The figure's font size. Default to 16.
    :param labels: Whether or not to display the column names. Defaults to the underlying data labels when there are
    50 columns or less, and no labels when there are more than 50 columns.
    :param sparkline: Whether or not to display the sparkline. Defaults to True.
    :param width_ratios: The ratio of the width of the matrix to the width of the sparkline. Defaults to `(15, 1)`.
    Does nothing if `sparkline=False`.
    :param color: The color of the filled columns. Default is `(0.25, 0.25, 0.25)`.
    :return: If `inline` is False, the underlying `matplotlib.figure` object. Else, nothing.
    """

    # only want to normalize on data, not market number or latitude/longitude
    # change index to market not 0,1,2,3... , 105
    if reindex:
        df = df.rename(index=df.iloc[:, 0])
        df = df.drop(df.columns[0], axis=1)
    # drop the latitude and longitude columns, helps with normalization
    if drop:
        df = df.drop(df.columns[-2:], axis=1)
    height = df.shape[0]
    width = df.shape[1]
    # n is True when df is nan.
    n = df.isnull()
    # z is the color-mask array, g is a NxNx3 matrix. Apply the z color-mask to set the RGB of each pixel.
    # maxval is the maximum value in the entire data frame
    maxval = df.max().max()
    # normalize dat data frame
    z = df / maxval
    # initilization of color NxNx3 matrix
    g = np.zeros((height, width, 3))
    g[n] = [1, 0.9, 0.9]  # washout red, if empty.
    # this starts at 0 level magnitudes and gets darker and darker depending on how larger the data point is.

    # not the most efficient way of doing it
    for i in range(0, 10):
        # remember g is a color matrix, in three dimensions. every point in the dataframe gets associated with a color!
        g[z > i / 10] = [(1 - i / 10), (1 - i / 10), (1 - i / 10)]
    # My additions end here
    # -------------------------------------------------------------------------------------------------------------- #
    # Set up the matplotlib grid layout. A unary subplot if no sparkline, a left-right splot if yes sparkline.
    fig = plt.figure(figsize=figsize)
    if sparkline:
        gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios)
        gs.update(wspace=0.08)
        ax1 = plt.subplot(gs[1])
    else:
        gs = gridspec.GridSpec(1, 1)
    ax0 = plt.subplot(gs[0])

    # Create the nullity plot.
    ax0.imshow(g, interpolation='none')

    # Remove extraneous default visual elements.
    ax0.set_aspect('auto')
    ax0.grid(b=False)
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
        ax0.set_xticklabels(list(df.columns), rotation=45, ha=ha, fontsize=fontsize)
    else:
        ax0.set_xticks([])

    # Adds Timestamps ticks if freq is not None, else set up the two top-bottom row ticks.
    if freq:
        ts_list = []

        if type(df.index) == pd.PeriodIndex:
            ts_array = pd.date_range(df.index.to_timestamp().date[0],
                                     df.index.to_timestamp().date[-1],
                                     freq=freq).values

            ts_ticks = pd.date_range(df.index.to_timestamp().date[0],
                                     df.index.to_timestamp().date[-1],
                                     freq=freq).map(lambda t:
                                                    t.strftime('%Y-%m-%d'))

        elif type(df.index) == pd.DatetimeIndex:
            ts_array = pd.date_range(df.index.date[0], df.index.date[-1],
                                     freq=freq).values

            ts_ticks = pd.date_range(df.index.date[0], df.index.date[-1],
                                     freq=freq).map(lambda t:
                                                    t.strftime('%Y-%m-%d'))
        else:
            raise KeyError('Dataframe index must be PeriodIndex or DatetimeIndex.')
        try:
            for value in ts_array:
                ts_list.append(df.index.get_loc(value))
        except KeyError:
            raise KeyError('Could not divide time index into desired frequency.')

        ax0.set_yticks(ts_list)
        ax0.set_yticklabels(ts_ticks, fontsize=20, rotation=0)

    else:
        # Sam made edits here.
        # loc_ticks is just a list of all numbers 0, ... , 105
        loc_ticks = list(range(0, df.shape[0] - 1))
        # we need 106 ticks
        ax0.set_yticks(loc_ticks)
        # label them with our market (index values)
        ax0.set_yticklabels(df.index.values, fontsize=6, rotation=0)
        # Sam's edits end here.

    # Create the inter-column vertical grid.
    in_between_point = [x + 0.5 for x in range(0, width - 1)]
    for in_between_point in in_between_point:
        ax0.axvline(in_between_point, linestyle='-', color='white')

    if sparkline:
        # Calculate row-wise completeness for the sparkline.
        completeness_srs = df.notnull().astype(bool).sum(axis=1)
        x_domain = list(range(0, height))
        y_range = list(reversed(completeness_srs.values))
        min_completeness = min(y_range)
        max_completeness = max(y_range)
        min_completeness_index = y_range.index(min_completeness)
        max_completeness_index = y_range.index(max_completeness)

        # Set up the sparkline, remove the border element.
        ax1.grid(b=False)
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
        ax1.plot(y_range, x_domain, color=color)

        if labels:
            # Figure out what case to display the label in: mixed, upper, lower.
            label = 'Data Completeness'
            if df.columns[0].islower():
                label = label.lower()
            if df.columns[0].isupper():
                label = label.upper()

            # Set up and rotate the sparkline label.
            ha = 'left'
            ax1.set_xticks([min_completeness + (max_completeness - min_completeness) / 2])
            ax1.set_xticklabels([label], rotation=45, ha=ha, fontsize=fontsize)
            ax1.xaxis.tick_top()
            ax1.set_yticks([])
        else:
            ax1.set_xticks([])
            ax1.set_yticks([])

        # Add maximum and minimum labels, circles.
        ax1.annotate(max_completeness,
                     xy=(max_completeness, max_completeness_index),
                     xytext=(max_completeness + 2, max_completeness_index),
                     fontsize=14,
                     va='center',
                     ha='left')
        ax1.annotate(min_completeness,
                     xy=(min_completeness, min_completeness_index),
                     xytext=(min_completeness - 2, min_completeness_index),
                     fontsize=14,
                     va='center',
                     ha='right')

        ax1.set_xlim([min_completeness - 2, max_completeness + 2])  # Otherwise the circles are cut off.
        ax1.plot([min_completeness], [min_completeness_index], '.', color=color, markersize=10.0)
        ax1.plot([max_completeness], [max_completeness_index], '.', color=color, markersize=10.0)

        # Remove tick mark (only works after plotting).
        ax1.xaxis.set_ticks_position('none')

    if inline:
        plt.show()
    else:
        return ax0


def geoplot_slr(df,
                filter = None, n =0, p = 0, sort = None, x = None,
                y = None, figsize=(25, 10), inline=True,
                by=None, cmap='YlGn', **kwargs):
    '''Function takes in a dataframe and plots a scatter plot of lat, long
         and then where the markets are on that plot, and color coded.'''
    lat = df.loc[:, 'LATITUDE'].tolist()
    long = df.loc[:, 'LONGITUDE'].tolist()
    # Don't look at last two rows
    df = df.iloc[:,:-2]
    df = nullity_filter(df, filter=filter, n=n, p=p)
    df = nullity_sort(df, sort=sort)

    # average out how many True values there are.
    notnull = (df.notnull().sum(axis='columns') / df.shape[1]).tolist()
    # high values of notnull are filled with values.
    # we plot lat long, notnull
    tuples = [(lat[i],long[i], notnull[i]) for i in range(0,len(lat))]
    plt.scatter(lat, long, c=notnull) # notnull is color scheme
    plt.colorbar()
    plt.show()
    return tuples
# Python code to sort a list of tuples
# according to given key.

# get the last key.
def last(n):
    return n[m]


# function to sort the tuple
def sort(tuples):
    # We pass used defined function last
    # as a parameter.
    return sorted(tuples, key=last)
m = 0