import example_geoplotslr as msno
import pandas as pd

df = pd.read_csv("geolocs.csv")
lat = df.loc[:, 'LATITUDE'].tolist()
long = df.loc[:, 'LONGITUDE'].tolist()
msno.matrix_new(df, reindex=True, drop = True)

tuples = msno.geoplot_slr(df)
m = 0
print("Sorted:")
print(msno.sort(tuples))
