from collections import OrderedDict
import copy
import os
import numpy as np
import seamless as ss


class KData():
    def __init__(self, df):
        self.df = df

    def get_targets(self, target_name):
        return self.df[target_name].squeeze()

    def get_mm(self, feature_names):
        return self.df.colslice(feature_names)


class KCVFitter():
    def __init__(self, model, save_dir=None):
        self.model = model
        self.save_dir = save_dir
        self._save_results = save_dir is not None
        if self._save_results:
            ss.io.ensure_dir_exists(self.save_dir)

    def fit(self, mm, targets, ixs, **fitting_kwargs):
        assert isinstance(ixs, OrderedDict)
        results = OrderedDict()
        early_stopping = 'early_stopping_rounds' in fitting_kwargs
        for fold_name, fold_ix in ixs.iteritems():
            train_ix, val_ix = fold_ix['train'], fold_ix['val']
            mm_train, targets_train = mm[train_ix], targets[train_ix]
            mm_val, targets_val = mm[val_ix], targets[val_ix]
            fold_model = copy.deepcopy(self.model)
            if early_stopping:
                fitting_kwargs['early_stopping_data'] = {
                        'mm': mm_val,
                        'targets': targets_val
                        }
            fold_model.fit(mm_train, targets_train, **fitting_kwargs)
            preds = fold_model.predict(mm_val)
            results[fold_name] = (fold_model, preds)

            if self._save_results:
                self._write_model(fold_model, fold_name)
                self._write_preds(fold_model, preds, fold_name)

        combined = np.concatenate([p for _, p in results.values()])
        results['combined_preds'] = combined
        return results

    def _write_model(self, model, fold_name):
        filename = 'model_' + self._get_path_suffix(model, fold_name)
        save_path = os.path.join(self.save_dir, filename)
        ss.io.update_latest_data(model, save_path)

    def _write_preds(self, model, preds, fold_name):
        filename = 'preds_' + self._get_path_suffix(model, fold_name)
        save_path = os.path.join(self.save_dir, filename)
        ss.io.update_latest_data(preds, save_path)

    def _get_path_suffix(self, model, fold_name):
        mtype = type(model).__name__
        desc = '{name}_{target}_fold-{fold}_{{timestamp}}.pkl'
        desc = desc.format(
                name=model.name,
                mtype=mtype,
                fold=fold_name
                )
        return desc
