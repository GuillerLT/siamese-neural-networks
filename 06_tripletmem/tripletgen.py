import keras
import numpy as np
import math


class tripletgen(keras.utils.Sequence):
    positive = 0.0
    negative = 1.0

    def __init__(self, mat_t, mat_m, trip, batch):
        self.mat_t = mat_t
        self.mat_m = mat_m
        self.trip = trip
        self.batch = batch
        self.n_k = mat_t.shape[0]
        self.n_s = mat_t.shape[1]
        self.shape = mat_t.shape[2:]
        self.emb_d = mat_m.shape[1]

    def __len__(self):
        return math.ceil(self.n_s / self.batch)

    def __getitem__(self, index):
        first = index * self.batch
        last = min(self.n_s, first + self.batch)
        size = (last - first) * self.n_k * self.trip
        x = {
            'i1': np.empty((size,) + self.shape),
            'm1': np.empty((size, self.emb_d)),
            'ia': np.empty((size,) + self.shape),
            'ma': np.empty((size, self.emb_d)),
            'i2': np.empty((size,) + self.shape),
            'm2': np.empty((size, self.emb_d)),
        }
        y = np.zeros((size, 2))
        i = 0
        for ja in range(first, last):
            for ka in range(self.n_k):
                xa = self.mat_t[ka][ja]
                for jp, kn in np.concatenate(
                        (
                            np.reshape(
                                np.random.choice(self.n_s, self.trip, False), (self.trip, 1)),
                            np.reshape(
                                np.random.choice(self.n_k, self.trip, False), (self.trip, 1)),
                        ), axis=1):
                    while jp == ja:
                        jp = np.random.choice(self.n_s)
                    kp = ka
                    xp = self.mat_t[kp][jp]
                    while kn == ka:
                        kn = np.random.choice(self.n_k)
                    jn = np.random.choice(self.n_s)
                    xn = self.mat_t[kn][jn]
                    if np.random.choice([True, False]):
                        x['i1'][i] = xp
                        x['m1'][i] = self.mat_m[kp]
                        y[i][0] = tripletgen.positive
                        x['ia'][i] = xa
                        x['ma'][i] = self.mat_m[ka]
                        y[i][1] = tripletgen.negative
                        x['i2'][i] = xn
                        x['m2'][i] = self.mat_m[kn]
                    else:
                        x['i1'][i] = xn
                        x['m1'][i] = self.mat_m[kn]
                        y[i][0] = tripletgen.negative
                        x['ia'][i] = xa
                        x['ma'][i] = self.mat_m[ka]
                        y[i][1] = tripletgen.positive
                        x['i2'][i] = xp
                        x['m2'][i] = self.mat_m[kp]
                    i += 1
        return x, y
