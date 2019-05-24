import matplotlib
import keras
import numpy as np
import datatool
import catgen
import cnn
from cnn import cnn_mnist, cnn_fashion_mnist, cnn_cifar10
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.utils import plot_model
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': 'CMU Serif'})
plt.rcParams.update({'font.size': 14})


# Parameters
dataset = cifar10
model = cnn_cifar10
n_tr = 100  # Number of train samples per class
batch = 32  # Number of samples per class in each epoch
optimizer = 'adam'
epochs = 90
# Data
mat_tr, mat_te = datatool.load(dataset, n_tr)
gen_tr = catgen.catgen(mat_tr, batch)
gen_te = catgen.catgen(mat_te, batch)
# Model
model = model(mat_tr.shape[0], mat_tr.shape[2:])
cnn.compile(model, optimizer)
# Train
model.fit_generator(gen_tr,
                    epochs=epochs,
                    verbose=0)
score, acc = model.evaluate_generator(gen_te)
print('Test score:', score)
print('Test accuracy:', acc)
