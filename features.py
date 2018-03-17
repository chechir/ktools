import numpy as np
import pandas as pd
from functools import partial

from ktools import tools


def lagged_ratio(df, groupby, calc_colname, fillna=-1):
    ids = df[groupby].values
    result = tools.group_apply(df[calc_colname].values, ids, _lagged_ratio)
    result = tools.fillna(result, fillna)
    return result


def _lagged_ratio(v, mask_nans=True):
    row_is_finite = np.isfinite(v)
    if mask_nans:
        v = tools.fillna(v, 0)
    cumsum = tools.lagged_cumsum(v, init=np.nan)
    n = tools.lagged_cumsum(row_is_finite, init=np.nan)
    result = cumsum/n
    return result


def lagged_values_by_group(df, col, groupby, shift, fillna=-1):
    lag_func = partial(tools.lag, init=np.nan, shift=shift)
    values = tools.group_apply(df[col].values, df[groupby].values, lag_func)
    values[~np.isfinite(values)] = fillna
    return values


def lagged_ratio_smoothing(df, col, groupby, smoothing_value=1,
                           min_samples_leaf=5, prior=0, condition_col=None):
    """ Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    if condition_col is None:
        lagged_ratios = lagged_ratio(df, groupby=groupby,
                                     calc_colname=col, fillna=prior)

    lagged_counts = count_by_group(df, groupby=groupby)
    smoothing = 1 / (1 + np.exp(-(lagged_counts - min_samples_leaf) / smoothing_value))
    result = prior * (1 - smoothing) + lagged_ratios * smoothing
    return result


def count_by_group(df, groupby):
    def counter(group_ids):
        return np.arange(len(group_ids)).astype('float')
    result = tools.group_apply(df[groupby].values, df[groupby].values, counter)
    return result


def ratio_smoothing(df_train, df_test, col, groupby, smoothing_value=1,
                    min_samples_leaf=5, prior=0,
                    smoothing=1):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """

    # Compute target mean
    averages = df_train[[groupby, col]].groupby(by=groupby)[col].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    # The bigger the count the less full_avg is taken into account
    averages[col] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    # from here and apply tdd
    # new_name = 'f:' + groupby
    df_test_result = pd.merge(
        df_test,
        averages.reset_index().rename(columns={'index': groupby, col: 'average'}),
        on=groupby,
        how='left')['average'].fillna(prior)
    # pd.merge does not keep the index so restore it
    df_test_result.index = df_test.index
    return df_test_result.values


def units_to():
    pass


def lagged_values_by_group_2dfs(df_train, df_test, col,
                                groupby, date_col, fillna=-1):
    df_test_temp = df_test[[date_col, groupby]]
    df_train_temp = df_train[[groupby, date_col, col]]
    # Compute last value for train_df
    max_dates = df_train[[groupby, date_col]].groupby(by=groupby)[date_col].agg(['max'])
    max_dates = max_dates.reset_index().rename(columns={'index': groupby, 'max': date_col})
    df_train_temp = max_dates.merge(df_train, how='inner', on=[groupby, date_col])
    df_train_temp = df_train_temp.groupby(by=[groupby, date_col])[col].agg(['mean'])
    df_train_temp = df_train_temp.reset_index().rename(columns={'mean': col})
    # Merge with test using the groupby col:
    df_test_result = df_test_temp.merge(df_train_temp, on=groupby, how='left').reset_index()
    df_test_result = df_test_result.fillna(fillna)
    return df_test_result[col].values

