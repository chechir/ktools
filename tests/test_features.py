from ktools import features
from ktools import tools
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


def test_lagged_values_by_group():
    df = pd.DataFrame({
        # 'year': [2012, 2013, 2014, 2015],
        'saleprice_log': [12, 14, 15, 3],
        'group': [1, 1, 1, 2]
        })
    result = features.lagged_values_by_group(df, 'saleprice_log', 'group', 1)
    expected = [-1, 12, 14, -1]
    assert(np.allclose(result, expected))


def test_lagged_values_two_dfs():
    ''' returns the features only for the test '''
    df_train = pd.DataFrame({
        'recordingdate': ['2012-01-01', '2012-02-02', '2012-02-03', '2012-02-03','2013-01-01'],
        'saleprice_log': [12, 14, 15, 17, 3],
        'group': [1, 1, 1, 1, 2]
        })
    df_train['recordingdate'] = pd.to_datetime(df_train['recordingdate'])
    df_test = pd.DataFrame({
        'recordingdate': ['2014-01-01', '2016-02-02', '2012-06-03'],
        'saleprice_log': [15, 3, -1],
        'group': [1, 2, 3]
        })
    result = features.lagged_values_by_group_2dfs(
            df_train, df_test,
            'saleprice_log',
            'group', 'recordingdate', fillna=-1)
    expected = [16, 3, -1]
    assert(np.allclose(result, expected))


def test_units_to_previous():
    v = np.array([2010, 2013, 2014, 2015, 2015, 2015])
    result = features.units_to_previous(v)
    expected = [-1, 3, 1, 1, 1, 1]
    assert(np.allclose(result, expected))


def test_grouped_units_to_previous():
    df = pd.DataFrame({
        'year': [2010, 2013, 2014, 2015, 2015, 2001],
        'group': [1, 1, 1, 1, 1, 2]
        })
    result = features.grouped_unit_to_previous(df, 'group', 'year')
    expected = [-1, 3, 1, 1, 1, -1]
    assert(np.allclose(result, expected))


def test_grouped_unit_to_previous_2dfs():
    ''' returns the features only for the test '''
    df_train = pd.DataFrame({
        'year': [2010, 2013, 2014, 2015, 2015, 2001],
        'group': [1, 1, 1, 1, 1, 2]
        })
    df_test = pd.DataFrame({
        'year': [2017, 2017, 2017],
        'group': [1, 2, 3]
        })
    result = features.grouped_unit_to_previous_2dfs(
            df_train, df_test, 'group', 'year', fillna=-1)
    expected = [2, 16, -1]
    assert(np.allclose(result, expected))
