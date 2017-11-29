import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pprint
import time
from itertools import product


def split_data(x, y, alfa):
    idx = np.random.permutation(len(x))
    xu = [x[i] for i in idx[0:int(np.floor(len(x) * alfa) + 1)]]
    yu = [y[i] for i in idx[0:int(np.floor(len(x) * alfa) + 1)]]
    xt = [x[i] for i in idx[int(np.floor(len(x) * alfa) + 1):len(x)]]
    yt = [y[i] for i in idx[int(np.floor(len(x) * alfa) + 1):len(x)]]
    return xu, yu, xt, yt


def points_gen_lin(n, w):
    X = np.random.rand(n, 2)
    y = [1 if X[i, 1] > X[i, 0] else 0 for i in range(len(X))]
    fuss = w * np.random.normal(0, 1, (n, 2))
    X = X + fuss
    for i in range(len(list(X))):
        if y[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], c="r")
        else:
            plt.scatter(X[i, 0], X[i, 1], c="g")
    plt.show()
    return X, y


def points_gen_chessboard(n, m, w):
    X = np.random.rand(n, 2)
    y = []
    for i in range(len(X)):
        for col, row in product(range(m), range(m)):
            if (row % 2 == 0 and col % 2 == 0) or (row % 2 == 1 and col % 2 == 1):
                if (row <= X[i, 0] * m < row + 1) and (col <= X[i, 1] * m < col + 1):
                    y.append(1)
                    break
            else:
                if (row <= X[i, 0] * m < row + 1) and (col <= X[i, 1] * m < col + 1):
                    y.append(0)
                    break
    fuss = w * np.random.normal(0, 1, (n, 2))
    X = X + fuss
    for i in range(len(list(X))):
        if y[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], c="r")
        else:
            plt.scatter(X[i, 0], X[i, 1], c="g")
    plt.show()
    return X, y


def knn_accuracy(k, X, y, attempts=1):
    accuracy = 0

    for i in range(attempts):
        xu, yu, xt, yt = split_data(X, y, 0.5)
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(xu, yu)
        accuracy += neigh.score(xt, yt)
    return accuracy/attempts

# def plot_knn_accuracy(points,step):
#     X,y = points_gen_lin(points,0.05)
#
#     lin_acc = []
#     for i in range(1, len(xt), step):
#         result.append((i, neigh.score(xt, yt)))
#
#
#
#
# start_time = time.time()
# X, y = points_gen_chessboard(1000, 5, 0.02)
# print("--- %s seconds ---" % (time.time() - start_time))


# xu, yu, xt, yt = split_data(X, y, 0.5)
#
# result = []
# for i in range(1, len(xt), 10):
#     neigh = KNeighborsClassifier(n_neighbors=i)
#     neigh.fit(xu, yu)
#     result.append((i, neigh.score(xt, yt)))
#
# pprint.pprint(result)
#
# plt.plot(np.array(result)[:, 0], np.array(result)[:, 1])
# plt.show()
