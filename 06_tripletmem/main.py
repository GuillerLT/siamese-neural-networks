import keras
import numpy as np
import datatool
import tripletgen
import base
import triplet
import predtool
from base import base_mnist, base_fashion_mnist, base_cifar10
from keras.datasets import mnist, fashion_mnist, cifar10
import sys

# Parameters
dataset = mnist
base_model = base_mnist
emb_d = 512
loss = triplet.contrastive_loss1
optimizer = 'adam'
n_tr = 100  # Number of train samples per class
trip = 3  # Number of triplets per each sample
batch = 16  # Number of samples per class in each epoch
epochs = 50
# Data
mat_tr, mat_te = datatool.load(dataset, n_tr)
mat_me = np.zeros((mat_tr.shape[0], emb_d))
gen_tr = tripletgen.tripletgen(mat_tr, mat_me, trip, batch)
gen_te = tripletgen.tripletgen(mat_te, mat_me, trip, batch)
# Model
base_model = base_model(mat_tr.shape[2:], emb_d)
trip_model, emb_model = triplet.create(base_model, emb_d)
triplet.compile(trip_model, loss, optimizer)
# Train
for epoch in range(epochs):
    print("Epoch {}".format(epoch + 1))
    loss = 0.0
    cont = 0
    for x, y in gen_tr:
        loss += trip_model.train_on_batch(x, y)
        cont += 1
    for i in range(mat_tr.shape[0]):
        new_me = np.average(emb_model.predict(mat_tr[i]), axis=0)
        mat_me[i] = mat_me[i] * (1.0 - epoch / epochs) + \
            new_me * (epoch / epochs)
    sys.stdout.flush()
x_tr = np.reshape(mat_tr, (-1,) + mat_tr.shape[2:])
e_tr = emb_model.predict(x_tr)
y_tr = np.repeat(np.arange(mat_tr.shape[0]), mat_tr.shape[1])
x_te = np.reshape(mat_te, (-1,) + mat_te.shape[2:])
e_te = emb_model.predict(x_te)
y_te = np.repeat(np.arange(mat_te.shape[0]), mat_te.shape[1])
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
