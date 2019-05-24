import keras
import numpy as np
import datatool
import pairgen
import base
import siamese
import predtool
from base import base_mnist, base_fashion_mnist, base_cifar10
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.callbacks import ModelCheckpoint

# Parameters
dataset = cifar10
base_model = base_cifar10
emb_d = 512
loss = siamese.contrastive_loss1
optimizer = 'adam'
n_tr = 100  # Number of train samples per class
pos = 1  # Number of positive pairs
neg = 4  # Number of negative pairs
batch = 32  # Number of samples per class in each epoch
epochs = 250
# Data
mat_tr, mat_te = datatool.load(dataset, n_tr)
gen_tr = pairgen.pairgen(mat_tr, pos, neg, batch)
gen_te = pairgen.pairgen(mat_te, pos, neg, batch)
# Model
base_model = base_model(mat_tr.shape[2:], emb_d)
siam_model, emb_model = siamese.create(base_model, mat_tr.shape[0])
siamese.compile(siam_model, loss, optimizer)
# Train
siam_model.fit_generator(gen_tr,
                         epochs=epochs,
                         verbose=2)
# Obtain embedded spaces
x_tr = np.reshape(mat_tr, (-1,) + mat_tr.shape[2:])
e_tr = emb_model.predict(x_tr)
y_tr = np.repeat(np.arange(mat_tr.shape[0]), mat_tr.shape[1])
x_te = np.reshape(mat_te, (-1,) + mat_te.shape[2:])
e_te = emb_model.predict(x_te)
y_te = np.repeat(np.arange(mat_te.shape[0]), mat_te.shape[1])
# Evaluate
predtool.hist(e_tr, y_tr, e_te, y_te)
predtool.svr(e_tr, y_tr, e_te, y_te, 'linear')
predtool.svr(e_tr, y_tr, e_te, y_te, 'poly')
predtool.svr(e_tr, y_tr, e_te, y_te, 'rbf')
predtool.knn(e_tr, y_tr, e_te, y_te, 1)
predtool.knn(e_tr, y_tr, e_te, y_te, 5)
predtool.knn(e_tr, y_tr, e_te, y_te, 15)
predtool.knn(e_tr, y_tr, e_te, y_te, 25)
predtool.rf(e_tr, y_tr, e_te, y_te, 50)
predtool.rf(e_tr, y_tr, e_te, y_te, 75)
predtool.rf(e_tr, y_tr, e_te, y_te, 100)
