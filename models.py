from ssmodels.ensemble import _LGBModel, _RegressorMixin
import lightgbm


class LGBMRegressorWrapper(_LGBModel, _RegressorMixin):
    _internal_model_cls = lightgbm.LGBMRegressor

    def predict(self, mm, *args, **kwargs):
        mm = mm[self._mm_columns]

        self._format_predict_kwargs(kwargs)
        predictions = self._predict(mm, *args, **kwargs)
        return predictions
