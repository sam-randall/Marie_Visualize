import pandas as pd
import matplotlib.pyplot as plt

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

def geoplot_slr(df,
                filter = None, n =0, p = 0, sort = None, x = None,
                y = None, figsize=(25, 10), inline=True,
                by=None, cmap='YlGn', **kwargs):
    lat = df.loc[:, 'LATITUDE'].tolist()
    long = df.loc[:, 'LONGITUDE'].tolist()

    df = df.iloc[:,:-2]
    df = nullity_filter(df, filter=filter, n=n, p=p)
    df = nullity_sort(df, sort=sort)


    notnull = (df.notnull().sum(axis='columns') / df.shape[1]).tolist()
    # high values of notnull are filled with values.

    tuples = [(lat[i],long[i], notnull[i]) for i in range(0,len(lat))]
    plt.scatter(lat, long, c=notnull)
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


# driver code


df = pd.read_csv("geolocs.csv")
lat = df.loc[:,'LATITUDE'].tolist()
long = df.loc[:,'LONGITUDE'].tolist()
tuples = geoplot_slr(df)
m = 0
print("Sorted:")
print(sort(tuples))
