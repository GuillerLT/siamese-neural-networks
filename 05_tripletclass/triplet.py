import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Concatenate, Dense


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


def create(base_model, n_k):
    i1 = Input(shape=base_model.input_shape[1:], name='i1')
    ia = Input(shape=base_model.input_shape[1:], name='ia')
    i2 = Input(shape=base_model.input_shape[1:], name='i2')

    ix = Input(shape=base_model.input_shape[1:])
    px = base_model(ix)
    cx = Dense(n_k, name='cx', activation='softmax')(px)
    c_model = Model(ix, cx)

    p1 = base_model(i1)
    pa = base_model(ia)
    p2 = base_model(i2)
    o1 = Lambda(euclidean_distance, name='o1')([pa, p1])
    o2 = Lambda(euclidean_distance, name='o2')([pa, p2])
    c1 = Dense(n_k, name='c1', activation='softmax')(p1)
    ca = Dense(n_k, name='ca', activation='softmax')(pa)
    c2 = Dense(n_k, name='c2', activation='softmax')(p2)
    o = Concatenate(name='o')([o1, o2])
    trip_model = Model([i1, ia, i2], [c1, ca, c2, o])
    emp_model = Model(ia, pa)
    return trip_model, emp_model


def compile(trip_model, loss, optimizer):
    loss = {
        'c1': 'categorical_crossentropy',
        'ca': 'categorical_crossentropy',
        'c2': 'categorical_crossentropy',
        'o':  loss,
    }
    loss_weights = {
        'c1': 0.001,
        'ca': 0.001,
        'c2': 0.001,
        'o':  1.0,
    }
    metrics = {
        'c1': ['accuracy'],
        'ca': ['accuracy'],
        'c2': ['accuracy'],
    }
    trip_model.compile(loss=loss, loss_weights=loss_weights,
                       optimizer=optimizer, metrics=metrics)
