import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Dense


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def contrastive_loss1(y_true, y_pred):
    margin = 1.0
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean((1 - y_true) * square_pred + y_true * margin_square)


def pair_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred > 0.5, y_true.dtype)))


def create(base_model, n_k):
    i1 = Input(shape=base_model.input_shape[1:], name='i1')
    i2 = Input(shape=base_model.input_shape[1:], name='i2')
    p1 = base_model(i1)
    p2 = base_model(i2)
    c1 = Dense(n_k, name='c1', activation='softmax')(p1)
    o = Lambda(euclidean_distance, name='o')([p1, p2])
    c2 = Dense(n_k, name='c2', activation='softmax')(p2)
    siam_model = Model([i1, i2], [c1, o, c2])
    emb_model = Model(i1, p1)
    return siam_model, emb_model


def compile(siam_model, loss, optimizer):
    loss = {
        'c1': 'categorical_crossentropy',
        'o':  loss,
        'c2': 'categorical_crossentropy',
    }
    contribution = 0.125
    loss_weights = {
        'c1': contribution,
        'o':  1.00,
        'c2': contribution,
    }
    metrics = {
        'c1': ['accuracy'],
        'o':  [pair_accuracy],
        'c2': ['accuracy'],
    }
    siam_model.compile(loss=loss,
                       loss_weights=loss_weights,
                       optimizer=optimizer,
                       metrics=metrics)
