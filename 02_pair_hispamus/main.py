import keras
import datatool
import pairgen
import base
import siamese
import predtool
from base import base_mnist, base_fashion_mnist, base_cifar10
from keras.datasets import mnist, fashion_mnist, cifar10
import numpy as np
import gc
from keras import backend as K
import sys

# Parameters
base = base_cifar10
emb_d = 256
loss = siamese.contrastive_loss1
optimizer = 'adam'
batch = 16  # Number of samples per class in each epoch
epochs = 200
for n in range(56):
    # Data
    x_tr, y_tr, x_te, y_te = datatool.load(n)
    gen_tr = pairgen.pairgen(x_tr, y_tr, batch)
    # Model
    base_model = base(x_tr.shape[1:], emb_d)
    siam_model, emb_model = siamese.create(base_model)
    siamese.compile(siam_model, loss, optimizer)
    # Train
    siam_model.fit_generator(gen_tr, epochs=epochs, verbose=0)
    # Obtain embedded spaces
    e_tr = emb_model.predict(x_tr)
    e_te = emb_model.predict(x_te)
    # Evaluate
    print("Imagen {:02}".format(n+1))
    predtool.hist(e_tr, y_tr, e_te, y_te)
    sys.stdout.flush()
    K.clear_session()
    gc.collect()
