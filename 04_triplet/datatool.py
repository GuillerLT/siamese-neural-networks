import keras
import numpy as np
import keras.backend as K


def normalize(x_t, y_t):
    channels = 1 if len(x_t.shape) == 3 else x_t.shape[3]
    if K.image_data_format() == 'channels_first':
        x_t = x_t.reshape(x_t.shape[0], channels, x_t.shape[1], x_t.shape[2])
        y_t = y_t.reshape((y_t.shape[0],))
    else:
        x_t = x_t.reshape(x_t.shape[0], x_t.shape[1], x_t.shape[2], channels)
        y_t = y_t.reshape((y_t.shape[0],))
    x_t = x_t.astype('float32')
    x_t /= 255.0
    return x_t, y_t


def matrix(x_t, y_t, n_t=None):
    keys = np.sort(np.unique(y_t))
    dict_t = {key: x_t[y_t == key] for key in keys}
    m_t = min(len(dict_t[key]) for key in keys)
    n_t = m_t if n_t is None else min(n_t, m_t)
    dict_t = {key: dict_t[key][:n_t] for key in keys}
    mat_t = np.concatenate([[dict_t[key]] for key in keys])
    return mat_t


def load(dataset, n_tr):
    (x_tr, y_tr), (x_te, y_te) = dataset.load_data()
    x_tr, y_tr = normalize(x_tr, y_tr)
    x_te, y_te = normalize(x_te, y_te)
    mat_tr = matrix(x_tr, y_tr, n_tr)
    mat_te = matrix(x_te, y_te)
    return mat_tr, mat_te
