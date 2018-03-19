from sklearn.ensemble import RandomForestRegressor as RFR
from collections import defaultdict
# from million._config import NULL_VALUE
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


@jit
def get_wale_loss(y, ypred):
    assert len(y) == len(ypred)
    n = len(y)
    value = np.sum([min(0.4, abs(y[i]-ypred[i])) for i in range(n)])
    return value / float(n)


def wale_lgb(truth, predictions):
    score = -1 * get_wale_loss(truth, predictions)
    return 'wale', score, True


# def get_test_ixs(targets):
#     ixs = targets == NULL_VALUE
#     return ixs


def ix_to_bool(ix, length):
    boolean_mask = np.repeat(False, length)
    boolean_mask[ix] = True
    return boolean_mask


def fillna(array, na_value):
    array = array.copy()
    ix = np.isnan(array) | np.isinf(array)
    if np.isscalar(na_value):
        array[ix] = na_value
    else:
        array[ix] = na_value[ix]
    return array


def group_apply(values, group_ids, func, multiarg=False, strout=False):
    if group_ids.ndim == 2:
        group_ids = add_as_strings(*[group_ids[:, i] for i in range(group_ids.shape[1])], sep='_')

    ix = np.argsort(group_ids, kind='mergesort')
    sids = group_ids[ix]
    cuts = sids[1:] != sids[:-1]
    reverse = invert_argsort(ix)
    values = values[ix]

    if strout:
        nvalues = np.prod(values.shape)
        res = np.array([None]*nvalues).reshape(values.shape)
    elif multiarg:
        res = np.nan * np.zeros(len(values))
    else:
        res = np.nan * np.zeros(values.shape)

    prevcut = 0
    for cut in np.where(cuts)[0]+1:
        if multiarg:
            res[prevcut:cut] = func(*values[prevcut:cut].T)
        else:
            res[prevcut:cut] = func(values[prevcut:cut])
        prevcut = cut
    if multiarg:
        res[prevcut:] = func(*values[prevcut:].T)
    else:
        res[prevcut:] = func(values[prevcut:])
    revd = res[reverse]
    return revd


def invert_argsort(argsort_ix):
    reverse = np.repeat(0, len(argsort_ix))
    reverse[argsort_ix] = np.arange(len(argsort_ix))
    return reverse


def add_as_strings(*args, **kwargs):
    result = args[0].astype(str)
    sep = kwargs.get('sep')
    if sep:
        seperator = np.repeat(sep, len(result))
    else:
        seperator = None

    for arr in args[1:]:
        if seperator is not None:
            result = _add_strings(result, seperator)
        result = _add_strings(result, arr.astype(str))
    return result


def _add_strings(v, w):
    return np.core.defchararray.add(v, w)


def lag(v, init, shift=1):
    w = np.nan * v
    w[0:shift] = init
    w[shift:] = v[:-shift]
    return w


def lagged_cumsum(v, init):
    return lag(np.cumsum(v, axis=0), init)


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


def ffill(values):
    assert len(values.shape) == 1 or values.shape[1] == 1, 'ffill only works for vector'
    values = np.atleast_2d(values)
    mask = is_null(values)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    idx = np.maximum.accumulate(idx, axis=1, out=idx)
    out = values[np.arange(idx.shape[0])[:, None], idx]
    out = out.reshape(-1,)
    return out


def is_null(*args, **kwargs):
    return pd.isnull(*args, **kwargs)


def bfill(values):
    reversed_values = values[::-1]
    reversed_values = ffill(reversed_values)
    values = reversed_values[::-1]
    return values


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
        params = sample_with_RFR(100, random_func, columns, results_path, metric=metric)
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
