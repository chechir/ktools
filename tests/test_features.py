from ktools import features
import pandas as pd
import numpy as np


def test_smoothed_ratio():
    df_train = pd.DataFrame({
        'saleprice_log': [12, 14, 15, 12.4],
        'group': [1, 1, 2, 3]
        })
    df_test = pd.DataFrame({
        'group': [1, 2, 3, 4]
        })
    result = features.ratio_smoothing(df_train, df_test, 'saleprice_log', 'group',
                                      prior=12.4, min_samples_leaf=1)
    expected = [12.83863515, 13.7, 12.4, 12.4]
    assert(np.allclose(result, expected))


def test_lagged_ratio_smoothing():
    df = pd.DataFrame({
        'year': [2012, 2013, 2014, 2015],
        'saleprice_log': [12, 14, 15, 3],
        'group': [1, 1, 1, 2]
        })
    result = features.lagged_ratio_smoothing(df, 'saleprice_log', 'group', prior=12.4,
                                             min_samples_leaf=1)
    expected = [12.4, 12.2, 12.83863515, 12.4]
    assert(np.allclose(result, expected))
