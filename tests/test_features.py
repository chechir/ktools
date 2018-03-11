from ktools import features
import pandas as pd
import numpy as np


def test_smoothed_ratio():
    df_train = pd.DataFrame({
        'saleprice_log': [12, 14, 15],
        'group': [1, 1, 2]
        })
    df_test = pd.DataFrame({
        'group': [1, 2, 3]
        })
    result = features.ratio_smoothing(df_train, df_test, 'saleprice_log', 'group',
                                      prior=12.4, min_samples_leaf=1)
    expected = [12.83863515, 13.7, 12.4]
    assert(np.allclose(result, expected))

