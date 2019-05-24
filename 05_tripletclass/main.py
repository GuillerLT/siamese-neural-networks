import keras
import numpy as np
import datatool
import tripletgen
import base
import triplet
import predtool
from base import base_mnist, base_fashion_mnist, base_cifar10
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.callbacks import ModelCheckpoint

# Parameters
dataset = cifar10
base_model = base_cifar10
emb_d = 512
loss = triplet.contrastive_loss1
optimizer = 'adam'
n_tr = 100  # Number of train samples per class
trip = 3  # Number of triplets per each sample
batch = 16  # Number of samples per class in each epoch
epochs = 300
# Data
mat_tr, mat_te = datatool.load(dataset, n_tr)
gen_tr = tripletgen.tripletgen(mat_tr, trip, batch)
gen_te = tripletgen.tripletgen(mat_te, trip, batch)
# Model
base_model = base_model(mat_tr.shape[2:], emb_d)
trip_model, emb_model = triplet.create(base_model, mat_tr.shape[0])
triplet.compile(trip_model, loss, optimizer)
# Train
trip_model.fit_generator(gen_tr,
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
predtool.rf(e_tr, y_tr, e_te, y_te)
predtool.knn(e_tr, y_tr, e_te, y_te)
