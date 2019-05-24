import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Concatenate


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def contrastive_loss1(y_true, y_pred):
    y_true1 = y_true[:, 0:1] * -2.0 + 1.0
    y_true2 = y_true[:, 1:2] * -2.0 + 1.0
    y_pred1 = y_pred[:, 0:1]
    y_pred2 = y_pred[:, 1:2]
    d1 = y_pred1 * y_true1
    d2 = y_pred2 * y_true2
    margin = 1.0
    loss = margin + d1 + d2
    return K.maximum(loss, 0.0)


def create(base_model):
    i1 = Input(shape=base_model.input_shape[1:], name='i1')
    ia = Input(shape=base_model.input_shape[1:], name='ia')
    i2 = Input(shape=base_model.input_shape[1:], name='i2')
    p1 = base_model(i1)
    pa = base_model(ia)
    p2 = base_model(i2)
    o1 = Lambda(euclidean_distance, name='o1')([pa, p1])
    o2 = Lambda(euclidean_distance, name='o2')([pa, p2])
    o = Concatenate()([o1, o2])
    trip_model = Model([i1, ia, i2], o)
    emp_model = Model(ia, pa)
    return trip_model, emp_model


def compile(trip_model, loss, optimizer):
    trip_model.compile(loss=loss, optimizer=optimizer)
