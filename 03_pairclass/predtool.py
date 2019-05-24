import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def hist(e_tr, y_tr, e_te, y_te):
    n_k = len(np.unique(y_tr))

    def preddiction(i_te, e_tr):
        def distance(i_tr, i_te):
            return np.linalg.norm(i_tr - i_te)
        return np.argmin(
            np.average(
                np.reshape(
                    np.apply_along_axis(distance, 1, e_tr, i_te),
                    (n_k, -1)
                ),
                1
            )
        )
        # distancias = np.reshape(np.apply_along_axis(distance, 1, e_tr, i_te),
        #                         (len(np.unique(y_te)), -1))
        # medias = np.average(distancias, 1)
        # indice = np.argmin(medias)
    p_te = np.apply_along_axis(preddiction, 1, e_te, e_tr)
    acc = (y_te == p_te).mean()
    print('Accuracy on test set (hist): %0.2f%%' % (100 * acc))


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


def knn(e_tr, y_tr, e_te, y_te, n_neighbors=100):
    clf = KNeighborsClassifier(n_neighbors)
    acc(clf, ' knn', e_tr, y_tr, e_te, y_te)
