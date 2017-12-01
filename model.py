import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from itertools import product


class Knn:
    def __init__(self):
        self.x = np.array([])
        self.y = []
        self.xu = []
        self.yu = []
        self.xt = []
        self.yt = []
        self.accuracy = 0
        self.accuracy_list = []

    def points_gen_lin(self, points_num, disorder_coeff):
        self.x = np.random.rand(points_num, 2)
        self.y = [1 if self.x[i, 1] > self.x[i, 0] else 0 for i in range(len(self.x))]
        disorder = disorder_coeff * np.random.normal(0, 1, (points_num, 2))
        self.x += disorder

    def points_gen_chessboard(self, points, squares, disorder_coeff):
        self.x = np.random.rand(points, 2)
        self.y = []
        for i in range(len(self.x)):
            for col, row in product(range(squares), range(squares)):
                if (row % 2 == 0 and col % 2 == 0) or (row % 2 == 1 and col % 2 == 1):
                    if (row <= self.x[i, 0] * squares < row + 1) and (col <= self.x[i, 1] * squares < col + 1):
                        self.y.append(1)
                        break
                else:
                    if (row <= self.x[i, 0] * squares < row + 1) and (col <= self.x[i, 1] * squares < col + 1):
                        self.y.append(0)
                        break
        disorder = disorder_coeff * np.random.normal(0, 1, (points, 2))
        self.x += disorder

    def split_data(self, alfa):
        idx = np.random.permutation(len(self.x))
        self.xu = [self.x[i] for i in idx[0:int(np.floor(len(self.x) * alfa) + 1)]]
        self.yu = [self.y[i] for i in idx[0:int(np.floor(len(self.x) * alfa) + 1)]]
        self.xt = [self.x[i] for i in idx[int(np.floor(len(self.x) * alfa) + 1):len(self.x)]]
        self.yt = [self.y[i] for i in idx[int(np.floor(len(self.x) * alfa) + 1):len(self.x)]]

    def knn_accuracy(self, k, attempts=20, split_ratio=0.5):
        self.accuracy = 0
        for i in range(attempts):
            self.split_data(split_ratio)
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(self.xu, self.yu)
            self.accuracy += neigh.score(self.xt, self.yt)
        self.accuracy /= attempts

    def knn_accuracy_list(self, step, split_ratio=0.5, attempts=20):
        self.split_data(split_ratio)
        self.accuracy_list = []
        for i in range(1, len(self.xt), step):
            self.knn_accuracy(i, attempts, split_ratio)
            self.accuracy_list.append((i, self.accuracy))

    def plot_data_set(self):
        for i in range(len(self.x)):
            if self.y[i] == 1:
                plt.scatter(self.x[i, 0], self.x[i, 1], c="r")
            else:
                plt.scatter(self.x[i, 0], self.x[i, 1], c="g")
        plt.title("Wizulizacja zestawu wygenerowanych punktów")
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

    @staticmethod
    def plot_knn_accuracy(accuracy_lists, labels, data_sets_num):
        for i in range(data_sets_num):
            plt.plot(np.array(accuracy_lists[i])[:, 0], np.array(accuracy_lists[i])[:, 1], label=labels[i])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                   ncol=3, fancybox=True, shadow=True)
        plt.xlabel("współczynnik k")
        plt.ylabel("dokładność")
        plt.show()

    @staticmethod
    def plot_decision_boundaries(X, y, n_neighbors):

        h = .002  # step size in the mesh

        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

        # we create an instance of Neighbours Classifier and fit the data.
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
        neigh.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 0.05, X[:, 0].max() + 0.05
        y_min, y_max = X[:, 1].min() - 0.05, X[:, 1].max() + 0.05
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = neigh.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("Granice decyzyjne i zestaw danych (k = %i)" % n_neighbors)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()
