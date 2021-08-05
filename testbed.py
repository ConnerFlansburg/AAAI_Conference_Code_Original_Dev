import numpy as np
import pandas as pd

# create a dataframe
d = {'x': [1, 2, 3], 'y': np.array([2, 4, 8]), 'z': 100}
df = pd.DataFrame(d)

with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
    print(df)
    print(f'\nDrop Yields\n {df.drop("y", axis=1)}')
    print(f'\niloc Yields\n {df.loc[:, df.columns != "y"]}')
    print(f'\niloc 2 Yields\n {df.loc[:, ["x", "z"]]}')
    print(f'\ndf[df.drop] Yields\n {df[df.columns.drop("y")]}')
    print('-----------------------------------------------------')
    print(f'\nNormal Indexing yields:\n{df["x"]}')

# https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
# https://blog.developerspoint.org/Linear-Regression-with-Scikit-Learn/
#
