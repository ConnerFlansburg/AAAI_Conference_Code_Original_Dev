# see: https://blog.developerspoint.org/Linear-Regression-with-Scikit-Learn/

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pathlib as pth
from sklearn.linear_model import LinearRegression, SGDRegressor, BayesianRidge, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import shuffle
import random
import matplotlib.pyplot as plt
from alive_progress import alive_bar

# TODO: expand this & comment

random.seed(239)
np.random.seed(239)


def scale_data(data: pd.DataFrame) -> pd.DataFrame:
    """ Scale the data using a MinMax Scalar. """

    # * Scale/Normalize the data using a MinMax Scalar * #
    model = MinMaxScaler()        # create the scalar
    model.fit(data)               # fit the scalar
    scaled = model.transform(data)  # transform the data
    # cast the scaled data back into a dataframe
    data = pd.DataFrame(scaled, index=data.index, columns=data.columns)

    return data


# read in
fl = str(pth.Path.cwd() / 'data' / 'kdfw_processed_data.csv')
# * Read in the Data * # (treating '********' as NaNs)
df: pd.DataFrame = pd.read_csv(fl, dtype=float, na_values='********',
                               parse_dates=['date'], index_col=0).sort_values(by='date')
# * Get rid of NaN & infinity * #
df = df.replace('********', np.nan).replace(np.inf, np.nan)
df = df.dropna(how='any', axis=1)

# * Convert the Date to an Int * #
# ? not having a date doesn't seem to reduce accuracy
df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
df['date'].astype(int)

df = scale_data(df)

print(list(df.min()))
print(list(df.max()))
d1 = df.min()
d2 = df.max()
d3 = pd.DataFrame({
    'Min': d1,  # make this a col
    'Max': d2   # make this a col
})
print(d3)

# split data
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
# ! This works fine
X1 = df.iloc[:, :-1].values                  # X = the attributes used to predict the target label
y = df.iloc[:, 1].values                    # Y = the label (OBS_tmpf_max) column
y = df['OBS_sknt_max'].values               # Y = the label (OBS_tmpf_max) column
# ! But this causes high error scores
X2 = df.drop('OBS_sknt_max', axis=1).values  # X = the attributes used to predict the target label
X3 = df[df.columns.drop('OBS_sknt_max')]

cols = list(df.columns)
cols.remove('OBS_sknt_max')
X_df = pd.DataFrame(X1, index=df.index, columns=cols)
# ! They should be equivalent
# ! X1 & X2 are NOT equal arrays:
# ! They do have the same size
# ! They do not have the same values
LABEL: str = 'GFS0_sktc_max'                          # the thing we are trying to predict
y = df[LABEL].values                                 # Y = the label (OBS_tmpf_max) column
# X = df.iloc[:, :-1].values                         # tutorial method  (mean sqr = 0.002548578892669875)
# X = df[df.columns.drop('OBS_sknt_max')].values     # other option (high error)
X = df[df.columns.drop(LABEL)].values     # other option (high error)
# X = df.drop('OBS_sknt_max', axis=1).values         # current  (high error)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

rslt = []
with alive_bar(100) as bar:
    for i in range(100):
        # split the data into test & train
        df = shuffle(df)                                  # randomly shuffle the array
        LABEL: str = 'GFS0_sktc_max'                         # the thing we are trying to predict
        y = df[LABEL].values                                 # Y = the label (OBS_tmpf_max) column
        X = df[df.columns.drop(LABEL)].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # run model
        # TODO: try different models here
        # lr = LinearRegression().fit(X_train, y_train)
        # lr = SGDRegressor().fit(X_train, y_train)
        lr = BayesianRidge().fit(X_train, y_train)
        # lr = Ridge().fit(X_train, y_train)
        # lr = Lasso().fit(X_train, y_train)

        # predict
        y_pred = lr.predict(X_test)

        dct = {
            'Iteration': [i],
            'Mean Absolute Error': [metrics.mean_absolute_error(y_test, y_pred)],
            'Mean Squared Error': [metrics.mean_squared_error(y_test, y_pred)],
            'Root Mean Squared Error': [np.sqrt(metrics.mean_squared_error(y_test, y_pred))]
        }

        f = pd.DataFrame(dct)

        rslt.append(f)
        bar()

frame = pd.concat(rslt)  # create a single dataframe from the results
frame.to_csv(str(pth.Path.cwd() / 'output' / 'testing.csv'))


def grid_plot(df: pd.DataFrame, file: str):
    """
    scatter_plot creates a scatter plot of the dataframe using
    Pandas libraries.
    """

    # create the figure that will hold all 4 plots
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    axes[0].ticklabel_format(style='sci', useMathText=True)
    axes[1].ticklabel_format(style='sci', useMathText=True)
    axes[2].ticklabel_format(style='sci', useMathText=True)

    # set the values for the 'Training Size' axis
    axes[0].set_ylabel('Error Score')  # label the y-axis
    axes[1].set_ylabel('Error Score')  # label the y-axis

    # axes[2].set_xticks(BUCKETS_LABEL)  # make a tick for every bucket
    axes[2].set_ylabel('Error Score')  # label the y-axis

    # rotate = 45  # how many degrees the x-axis labels should be rotated
    rotate = 90  # how many degrees the x-axis labels should be rotated
    # create the plot & place it in the upper left corner
    df.plot(ax=axes[0],
            kind='line',
            x='Iteration',
            y='Mean Absolute Error',
            color='blue',
            style='--',      # the line style
            x_compat=True,
            rot=rotate,      # how many degrees to rotate the x-axis labels
            # use_index=True,
            grid=True,
            legend=True,
            # marker='o',    # what type of data markers to use?
            # mfc='black'    # what color should they be?
            )
    # axes[0].set_title('Mean Absolute Error')

    # create the plot & place it in the upper right corner
    df.plot(ax=axes[1],
            kind='line',
            x='Iteration',
            y='Mean Squared Error',
            color='red',
            style='-.',    # the line style
            rot=rotate,    # how many degrees to rotate the x-axis labels
            x_compat=True,
            # use_index=True,
            grid=True,
            legend=True,
            # marker='o',  # what type of data markers to use?
            # mfc='black'  # what color should they be?
            )
    # axes[1].set_title('Mean Squared Error')

    # create the plot & place it in the lower left corner
    df.plot(ax=axes[2],
            kind='line',
            x='Iteration',
            y='Root Mean Squared Error',
            color='green',
            style='-',     # the line style
            rot=rotate,    # how many degrees to rotate the x-axis labels
            x_compat=True,
            # use_index=True,
            grid=True,
            legend=True,
            # marker='o',  # what type of data markers to use?
            # mfc='black'  # what color should they be?
            )
    # axes[2].set_title('Mean Signed Error')

    fig.tight_layout()

    # save the plot to the provided file path
    plt.savefig(file)
    # show the plot
    plt.show()

    return


grid_plot(frame, str(pth.Path.cwd() / 'output' / 'testing.png'))


