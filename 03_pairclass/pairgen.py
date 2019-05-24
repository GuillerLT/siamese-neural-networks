import keras
import numpy as np
import math


class pairgen(keras.utils.Sequence):
    positive = 0.0
    negative = 1.0

    def __init__(self, mat_t, pos, neg, batch):
        self.mat_t = mat_t
        self.pos = pos
        self.neg = neg
        self.batch = batch
        self.n_k = mat_t.shape[0]
        self.n_s = mat_t.shape[1]
        self.shape = mat_t.shape[2:]

    def __len__(self):
        return math.ceil(self.n_s / self.batch)

    def __getitem__(self, index):
        first = index * self.batch
        last = min(self.n_s, first + self.batch)
        size = (last - first) * self.n_k * (self.pos + self.neg)
        x = {
            'i1': np.zeros((size,) + self.shape),
            'i2': np.zeros((size,) + self.shape),
        }
        y = {
            'c1': np.zeros((size, self.n_k)),
            'o':  np.zeros((size, )),
            'c2': np.zeros((size, self.n_k)),
        }
        i = 0
        for j1 in range(first, last):
            for k1 in range(self.n_k):
                x1 = self.mat_t[k1][j1]
                for j2 in np.random.choice(self.n_s, self.pos, False):
                    while j1 == j2:
                        j2 = np.random.choice(self.n_s)
                    x2 = self.mat_t[k1][j2]
                    if np.random.choice([True, False]):
                        x['i1'][i] = x1
                        x['i2'][i] = x2
                    else:
                        x['i1'][i] = x2
                        x['i2'][i] = x1
                    y['c1'][i][k1] = 1.0
                    y['o'][i] = pairgen.positive
                    y['c2'][i][k1] = 1.0
                    i += 1
                for k2 in np.random.choice(self.n_k, self.neg, False):
                    while k1 == k2:
                        k2 = np.random.choice(self.n_k)
                    j2 = np.random.choice(self.n_s)
                    x2 = self.mat_t[k2][j2]
                    if np.random.choice([True, False]):
                        x['i1'][i] = x1
                        x['i2'][i] = x2
                        y['c1'][i][k1] = 1.0
                        y['c2'][i][k2] = 1.0
                    else:
                        x['i1'][i] = x2
                        x['i2'][i] = x1
                        y['c1'][i][k2] = 1.0
                        y['c2'][i][k1] = 1.0
                    y['o'][i] = pairgen.negative
                    i += 1
        return x, y
