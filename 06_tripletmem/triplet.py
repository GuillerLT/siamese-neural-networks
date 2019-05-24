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
    d_pred1 = y_pred[:, 2:3]
    d_preda = y_pred[:, 3:4]
    d_pred2 = y_pred[:, 4:5]
    o1 = y_pred1 * y_true1
    o2 = y_pred2 * y_true2
    margin = 1.0
    loss = margin + o1 + o2
    loss += d_pred1 * 0.01
    loss += d_preda * 0.01
    loss += d_pred2 * 0.01
    return K.maximum(loss, 0.0)


def create(base_model, emb_d):
    i1 = Input(shape=base_model.input_shape[1:], name='i1')
    m1 = Input(shape=(emb_d,), name='m1')
    ia = Input(shape=base_model.input_shape[1:], name='ia')
    ma = Input(shape=(emb_d,), name='ma')
    i2 = Input(shape=base_model.input_shape[1:], name='i2')
    m2 = Input(shape=(emb_d,), name='m2')
    p1 = base_model(i1)
    pa = base_model(ia)
    p2 = base_model(i2)
    o1 = Lambda(euclidean_distance, name='o1')([pa, p1])
    o2 = Lambda(euclidean_distance, name='o2')([pa, p2])
    d1 = Lambda(euclidean_distance)([p1, m1])
    da = Lambda(euclidean_distance)([pa, ma])
    d2 = Lambda(euclidean_distance)([p2, m2])
    o = Concatenate()([o1, o2, d1, da, d2])
    trip_model = Model([i1, ia, i2, m1, ma, m2], o)
    emp_model = Model(ia, pa)
    return trip_model, emp_model


def compile(trip_model, loss, optimizer):
    trip_model.compile(loss=loss, optimizer=optimizer)
