from sklearn.ensemble import RandomForestRegressor as RFR
from collections import defaultdict
from million._config import NULL_VALUE
import cPickle
import json
import numpy as np
import pandas as pd
import os
import seamless as ss
import logging
from numba import jit


def ensemble_preds(predictions, weights):
    return np.average(predictions, weights=weights, axis=0)


def get_mae_loss(y, ypred):
    assert len(y) == len(ypred)
    n = len(y)
    return np.sum([abs(y[i]-ypred[i]) for i in range(n)]) / float(n)


def get_wale_loss(y, ypred):
    assert len(y) == len(ypred)
    n = len(y)
    value = np.sum([min(0.4, abs(y[i]-ypred[i])) for i in range(n)])
    return value / float(n)


def get_test_ixs(targets):
    ixs = targets == NULL_VALUE
    return ixs


def ix_to_bool(ix, length):
    boolean_mask = np.repeat(False, length)
    boolean_mask[ix] = True
    return boolean_mask


def get_group_ixs(*group_ids, **kwargs):
    """ Returns a dictionary {groupby_id: group_ix}.

    group_ids:
        List of IDs to groupbyy
    kwargs:
        bools = True or False, if True returns a boolean array
    """
    group_ids = _ensure_group_ids_hashable(group_ids)
    grouped_ixs = _get_group_ixs(group_ids)
    grouped_ixs = _convert_int_indices_to_bool_indices_if_necessary(grouped_ixs, kwargs)
    return grouped_ixs


def _ensure_group_ids_hashable(group_ids):
    if len(group_ids) == 1:
        combined_group_ids = group_ids[0]
    else:
        combined_group_ids = zip(*group_ids)

    is_list_of_list = lambda ids: isinstance(ids[0], list)
    is_matrix = lambda ids: isinstance(ids, np.ndarray) and ids.ndim == 2
    if is_list_of_list(combined_group_ids) or is_matrix(combined_group_ids):
        hashable_group_ids = [tuple(group_id) for group_id in combined_group_ids]
    else:
        hashable_group_ids = combined_group_ids

    return hashable_group_ids


def _convert_int_indices_to_bool_indices_if_necessary(ixs, kwargs):
    bools = kwargs.get('bools', False)
    if bools:
        length = np.sum([len(v) for v in ixs.itervalues()])
        ix_to_bool = lambda v, length: ix_to_bool(v, length)
        ixs = {k: ix_to_bool(v, length) for k, v in ixs.iteritems()}
    return ixs


def get_logger(file_name):
    logger_format = '%(asctime)s - %(filename)s - %(message)s'
    logging.basicConfig(filename=file_name, level=logging.DEBUG, format=logger_format)
    logger = logging.getLogger()
    return logger


def _get_group_ixs(ids):
    """ Returns a dictionary {groupby_id: group_ix}.

    ** Code Hall Of Fame **.
    """
    id_hash = defaultdict(list)
    for j, key in enumerate(ids):
        id_hash[key].append(j)
    id_hash = {k: np.array(v) for k, v in id_hash.iteritems()}
    return id_hash


def read_pickle(path):
    with open(path, 'rb') as f:
        obj = cPickle.load(f)
    return obj


def write_pickle(obj, path):
    with open(path, 'wb') as f:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)


def dropbox():
    path = os.path.expanduser('~/Dropbox/')
    if not os.path.isdir(path):
        path = '/Dropbox/'
    return path


def experiments():
    dropbox_path = dropbox()
    path = dropbox_path + 'experiments/'
    return path


def write_results_to_json(results_dict, path):
    with open(path, 'a') as f:
        json_format_data = json.dumps(results_dict)
        f.write(json_format_data + '\n')


def read_special_json(path):
    with open(path, 'r') as f:
        raw = f.read()
    raw_data = raw.split('\n')
    data = []
    for d in raw_data:
        if d:
            data.append(json.loads(d))
    return ss.DDF(data)


def convert_to_python_types(dic):
    for key in dic:
        dic[key] = dic[key].item()
    return dic


def rank_ensebmle(preds, weights):
    predictions = []
    for pred in preds:
        predictions.append(pd.Series(pred).rank().values / float(len(pred)))
    ens_predictions = np.average(predictions, weights=weights, axis=0)
    ens_predictions = np.clip(ens_predictions, 0, 1)
    return ens_predictions


@jit
def eval_gini(y_true, y_prob):
    """
    Original author CPMP : https://www.kaggle.com/cpmpml
    In kernel : https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini


def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = eval_gini(labels, preds)
    return [('gini', gini_score)]


def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = eval_gini(y, preds) / float(eval_gini(y, y))
    return 'gini', score, True


def sample_params(random_func, columns, results_path, random=False, metric='mae'):
    if random:
        params = random_func()
    else:
        params = sample_with_RFR(70, random_func, columns, results_path, metric=metric)
    return params


def sample_with_RFR(num_elements, random_func, columns, results_path, metric='mae'):
    # TODO: make sure this thing does one thing only
    random_data = [random_func() for _ in range(num_elements)]
    random_df = ss.DDF(random_data)
    csv_df = ss.DDF.from_csv(results_path)
    mm_train = csv_df[columns]
    targets = csv_df[metric]
    model = RFR(n_estimators=70)
    model.fit(mm_train, targets)
    predicted_losses = model.predict(random_df[columns])
    best_prediction_ix = np.argmin(predicted_losses)
    result = random_df.rowslice(best_prediction_ix)
    result = result.to_dict()
    result = {k: v[0] for k, v in result.items()}
    if 'has_time' in result:  # catboost
        result['has_time'] = bool(result['has_time'])
        result['verbose'] = bool(result['verbose'])
        result['use_best_model'] = bool(result['use_best_model'])
    print 'Expected Loss: {}'.format(predicted_losses[best_prediction_ix])
    return result


def categorical_to_numeric(df, column):
    def char_to_numeric(char):
        return str(ord(char))

    def text_to_numeric(text):
        text = str(text).strip()
        text = text[:10]
        text = text.lower()
        numeric_chars = map(char_to_numeric, text)
        result = ''.join(numeric_chars)
        result = float(result)
        return result

    result = map(text_to_numeric, df[column])
    result = np.log(np.array(result))
    return result


def log_transform(targets):
    return np.log(targets + 1)


def inv_log_transform(targets):
    return np.exp(targets) - 1


