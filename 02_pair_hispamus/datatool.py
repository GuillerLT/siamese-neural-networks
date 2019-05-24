import keras
import numpy as np
import keras.backend as K

import csv
import cv2
from PIL import Image


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


def load(n):
    y = dict()
    x_tr = []
    y_tr = []
    x_te = []
    y_te = []
    for data in csv.reader(open('data.txt', 'r'), delimiter=';'):
        image = np.fromstring(data[3], sep=',')
        image = np.reshape(image, (int(data[1]), int(data[2])))
        canvas = np.full((200, 200), 255.0)
        v = (200 - image.shape[0]) // 2
        h = (200 - image.shape[1]) // 2
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                canvas[j + h, i + v] = image[i, j]
        canvas = cv2.resize(canvas, (200, 200))
        x_tr += [canvas]
        if data[0] not in y:
            y[data[0]] = len(y)
            if n == 0:
                print("{}: {}".format(y[data[0]], data[0]))
        y_tr += [y[data[0]]]
        # Image.fromarray(canvas).convert('RGB').save(
        #     "images/{}{}-{:02}.png".format(data[0], y[data[0]], len(y_tr)))
    x_te = x_tr[n:n+1]
    y_te = y_tr[n:n+1]
    x_tr = x_tr[0:n] + x_tr[n+1:len(x_tr)]
    y_tr = y_tr[0:n] + y_tr[n+1:len(y_tr)]
    x_tr = np.array(x_tr)
    y_tr = np.array(y_tr)
    x_te = np.array(x_te)
    y_te = np.array(y_te)
    x_tr, y_tr = normalize(x_tr, y_tr)
    x_te, y_te = normalize(x_te, y_te)
    return x_tr, y_tr, x_te, y_te
