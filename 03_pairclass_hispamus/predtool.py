import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from PIL import Image


def hist1(e_tr, y_tr, e_te, y_te):
    acc = 0
    for i in range(len(e_te)):
        res = {y: 0.0 for y in np.unique(y_tr)}
        tot = {y: 0 for y in np.unique(y_tr)}
        for j in range(len(e_tr)):
            res[y_tr[j]] += np.linalg.norm(e_tr[j] - e_te[i])
            tot[y_tr[j]] += 1
        best_clase = None
        best_dist = np.inf
        for clase in res:
            if tot[clase] != 0:
                dist = res[clase] / tot[clase]
                if dist < best_dist:
                    best_clase = clase
                    best_dist = dist
        if best_clase == y_te[i]:
            print("Prediccion: {}".format(best_clase))
            print("Real      : {}".format(y_te[i]))
            acc += 1
        else:
            print("Prediccion: {}".format(best_clase))
            print("Real      : {}".format(y_te[i]))
    print('Accuracy on test set (hist1): %0.2f%%' % (100 * acc / len(e_te)))


def hist2(x_tr, e_tr, y_tr, x_te, e_te, y_te):
    acc = 0
    for i in range(len(e_te)):
        low_ind = None
        low_dis = np.inf
        for j in range(len(e_tr)):
            dis = np.linalg.norm(e_tr[j] - e_te[i])
            if dis < low_dis:
                low_ind = j
                low_dis = dis
        low_class = y_tr[low_ind]
        print("Prediccion: {} ({})".format(low_class, low_ind+1))
        print("Real      : {}".format(y_te[i]))
        if low_class == y_te[i]:
            acc += 1
    print('Accuracy on test set (hist2): %0.2f%%' % (100 * acc / len(e_te)))


def acc(clf, str, e_tr, y_tr, e_te, y_te):
    clf.fit(e_tr, y_tr)
    acc = clf.score(e_te, y_te)
    print('Accuracy on test set ({}): %0.2f%%'.format(str) % (100 * acc))


def svr(e_tr, y_tr, e_te, y_te, kernel='rbf'):
    clf = SVR(kernel=kernel, gamma='auto')
    acc(clf, ' svr', e_tr, y_tr, e_te, y_te)


def rf(e_tr, y_tr, e_te, y_te, n_estimators=100):
    clf = RandomForestClassifier(n_estimators=n_estimators)
    acc(clf, '  rf', e_tr, y_tr, e_te, y_te)


def knn(e_tr, y_tr, e_te, y_te, n_neighbors=1):
    clf = KNeighborsClassifier(n_neighbors)
    acc(clf, ' knn', e_tr, y_tr, e_te, y_te)
