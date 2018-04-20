import numpy as np
from collections import OrderedDict

from ktools.modelling import KCVFitter, KData
from ktools.models import LGBMRegressorWrapper
from ddf import DDF


def get_small_df():
    df = DDF({
            'col1': np.arange(10),
            'target': np.append(np.zeros(5), np.ones(5))
            })
    return df


def test_KCVFitter():
    df = get_small_df()
    data = KData(df)
    mm, targets = data.get_mm(['col1']), data.get_targets('target')
    model = LGBMRegressorWrapper({'n_estimators': 1})
    fitter = KCVFitter(model)
    ixs = OrderedDict()
    train_ixs = targets == 0
    ixs[0] = {'train': train_ixs, 'val': ~train_ixs}
    result = fitter.fit(mm, targets, ixs)
    preds = result['combined_preds']
    assert len(preds) == 5
