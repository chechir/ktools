import numpy as np
from functools import partial

from ktools import tools


def lagged_ratio(df, groupby, calc_colname, fillna=-1):
    ids = df[groupby]
    result = tools.group_apply(df[calc_colname], ids, _lagged_ratio)
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


def lagged_values_by_group(df, groupby, calc_colname, shift, fillna=-1):
    lag_func = partial(tools.lag, init=np.nan, shift=shift)
    values = tools.group_apply(df[calc_colname], df[groupby], lag_func)
    values[~np.isfinite(values)] = fillna
    return values


def lagged_ratio_smoothing(df, col, groupby, smoothing_value=1,
                           min_samples_leaf=5, prior=0, condition_col=None):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    if condition_col is None:
        lagged_ratios = lagged_ratio(df, groupby=groupby,
                                     calc_colname=col, fillna=0)

    lagged_counts = count_by_group(df, groupby=groupby)
    smoothing = 1 / (1 + np.exp(-(lagged_counts - min_samples_leaf) / smoothing_value))
    result = prior * (1 - smoothing) + lagged_ratios * smoothing
    return result


def count_by_group(df, groupby):
    def counter(group_ids):
        return np.arange(len(group_ids)).astype('float')
    result = np.group_apply(df['win_flag'], df[groupby], counter)
    return result
