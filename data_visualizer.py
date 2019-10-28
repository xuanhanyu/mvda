from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold.t_sne import TSNE
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
matplotlib.use('GTK3Agg')


class DataVisualizer:
    cmaps = ['red', 'green', 'blue', 'yellow', 'cyan', 'orange', 'purple', 'teal', 'steelblue', 'crimson', 'pink',
             'navy']
    markers = ['o', '^', 's', '*', 'p', 'P', 'v', 'X', 'D', 'H', "2", '$...$']
    linewidth = 0.1
    alpha = 0.5

    def __init__(self, algo=TSNE):
        self.manifold = algo
        self.scatter_params = {'linewidth': self.linewidth, 'alpha': self.alpha}
        self.pausing = False
        self.fig = self.ax = None

    def pdf(self, *args, transform):
        plt.rc('grid', linestyle="dotted", color='black', alpha=0.6)
        pts = list(itertools.product(*args))
        pts = np.array(pts)
        print(len(pts), pts.shape)
        transformed = transform(pts).squeeze()
        print(len(transformed), transformed.shape)
        pdfs = norm.pdf(transformed)
        dim = len(args)
        transformed, pdfs = map(list, zip(*sorted(zip(transformed, pdfs))))
        fig, ax = plt.subplots()
        ax.plot(transformed, pdfs, 'go--', lw=1, alpha=0.8, label='PDF')
        plt.show()

    def scatter(self, X, ys=None):
        if len(X) > 0:
            plt.rc('grid', linestyle="dotted", color='black', alpha=0.6)
            dim = X[0].shape[1]
            labels = ys if ys is not None else np.arange(len(X))
            if dim <= 2:
                if dim == 1:
                    tmp = [np.array([[x, 0] for x in X_i]) for X_i in X]
                    X = tmp
                fig, ax = plt.subplots()
                for i, X_i in enumerate(X):
                    ax.scatter(X_i[:, 0], X_i[:, 1], c=self.cmaps[i],
                               label=labels[i], **self.scatter_params)
            elif dim == 3:
                fig = plt.figure()
                ax = Axes3D(fig)
                for i, X_i in enumerate(X):
                    ax.scatter(X_i[:, 0], X_i[:, 1], X_i[:, 2], c=self.cmaps[i],
                               label=labels[i], **self.scatter_params)
            else:
                raise AttributeError('Unable to plot space of dimension greater than 3!')
            ax.legend()
            ax.set_title('{}D feature space'.format(dim))
            plt.grid(True)
            # plt.show()

    def mv_scatter(self, Xs, ys=None, title=None):
        dims = [Xs[0].shape[1] for Xs in Xs]
        max_dim = max(dims)
        if not self.pausing:
            if max_dim <= 2:
                self.fig, self.ax = plt.subplots()
                if max_dim == 1:
                    max_dim = 2
            elif max_dim == 3:
                self.fig = plt.figure()
                self.ax = Axes3D(self.fig)
            else:
                raise AttributeError('Unable to plot space of dimension greater than 3!')
            plt.rc('grid', linestyle="dotted", color='black', alpha=0.6)
        else:
            self.ax.clear()
        if title is not None:
            self.fig.canvas.set_window_title(title)

        for v in range(len(Xs)):
            labels = ys if ys is not None else np.arange(len(Xs[v]))
            if dims[v] < max_dim:
                Xs[v] = [np.array([np.concatenate([x, np.zeros(max_dim - dims[v])], axis=0) for x in X_i]) for X_i in
                         Xs[v]]
            if max_dim <= 2:
                for i, X_i in enumerate(Xs[v]):
                    self.ax.scatter(X_i[:, 0], X_i[:, 1], c=self.cmaps[i % 12], marker=self.markers[v % 12],
                                    s=50, label=labels[i], **self.scatter_params)
            elif max_dim == 3:
                for i, X_i in enumerate(Xs[v]):
                    self.ax.scatter(X_i[:, 0], X_i[:, 1], X_i[:, 2], c=self.cmaps[i], marker=self.markers[v % 12],
                                    s=50, label=labels[i], **self.scatter_params)
            else:
                raise AttributeError('Unable to plot space of dimension greater than 3!')
        self.ax.legend()
        self.ax.set_title('{}D feature space'.format(dims))
        plt.grid(True)

    def pause(self, interval=0.001):
        self.pausing = True
        plt.pause(interval)

    def show(self, block=True):
        self.pausing = False
        plt.show(block=block)
