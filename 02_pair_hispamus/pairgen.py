import keras
import numpy as np
import math


class pairgen(keras.utils.Sequence):
    positive = 0.0
    negative = 1.0

    def __init__(self, x_tr, y_tr, batch):
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.batch = batch
        self.shape = x_tr.shape[1:]

    def __len__(self):
        return math.ceil(len(self.x_tr) / self.batch)

    def __getitem__(self, index):
        if index == 0:
            permutation = np.random.permutation(len(self.y_tr))
            self.x_tr = self.x_tr[permutation]
            self.y_tr = self.y_tr[permutation]
        first = index * self.batch
        last = min((math.factorial(len(self.x_tr)) /
                    (math.factorial(2) * math.factorial(len(self.x_tr) - 2))),
                   first + self.batch)
        size = (last - first)
        x = {
            'i1': np.empty((size,) + self.shape),
            'i2': np.empty((size,) + self.shape),
        }
        y = np.zeros((size, ))
        n = 0
        for i in range(first, last):
            j = 1
            k = len(self.x_tr) - 1
            while i >= k:
                i -= k
                j += 1
                k -= 1
            if np.random.choice([True, False]):
                x['i1'][n] = self.x_tr[i]
                x['i2'][n] = self.x_tr[i + j]
            else:
                x['i1'][n] = self.x_tr[i + j]
                x['i2'][n] = self.x_tr[i]
            if self.y_tr[i] == self.y_tr[i + j]:
                y[n] = pairgen.positive
            else:
                y[n] = pairgen.negative
            n += 1
        return x, y
