import keras
import numpy as np
import math


class catgen(keras.utils.Sequence):
    def __init__(self, mat_t, batch):
        self.mat_t = mat_t
        self.batch = batch
        self.n_k = mat_t.shape[0]
        self.n_s = mat_t.shape[1]
        self.shape = mat_t.shape[2:]

    def __len__(self):
        return math.ceil(self.n_s / self.batch)

    def __getitem__(self, index):
        first = index * self.batch
        last = min(self.n_s, first + self.batch)
        size = (last - first) * self.n_k
        x = np.empty((size,) + self.shape)
        y = np.zeros((size, self.n_k))
        i = 0
        for j in range(first, last):
            for k in range(self.n_k):
                x[i] = self.mat_t[k][j]
                y[i][k] = 1.0
                i += 1
        return x, y
