from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
import torch

matplotlib.use('GTK3Agg')


class DataVisualizer:
    cmaps = ['red', 'green', 'blue', 'yellow', 'cyan', 'orange', 'purple', 'teal', 'steelblue', 'crimson', 'pink',
             'navy']
    markers = ['o', '^', 's', '*', 'p', 'P', 'v', 'X', 'D', 'H', "2", '$...$']
    linewidth = 0.1
    alpha = 0.5

    def __init__(self, embed_algo=None, embed_style='global', grid=True, legend=True):
        if embed_algo is not None and (hasattr(embed_algo, 'n_components') and embed_algo.n_components > 3):
            raise AssertionError
        assert embed_algo is None or embed_style in ['per_view', 'global']
        self.manifold = embed_algo
        self.manifold_style = embed_style
        self.scatter_params = {'linewidth': self.linewidth, 'alpha': self.alpha}
        self.pausing = False
        self.grid = grid
        self.legend = legend
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
                    ax.scatter(X_i[:, 0], X_i[:, 1],
                               c=self.cmaps[i % len(self.cmaps)],
                               label=labels[i], **self.scatter_params)
            elif dim == 3:
                fig = plt.figure()
                ax = Axes3D(fig)
                for i, X_i in enumerate(X):
                    ax.scatter(X_i[:, 0], X_i[:, 1], X_i[:, 2],
                               c=self.cmaps[i % len(self.cmaps)],
                               label=labels[i], **self.scatter_params)
            else:
                raise AttributeError('Unable to plot space of dimension greater than 3!')
            if self.legend: ax.legend()
            ax.set_title('{}D feature space'.format(dim))
            plt.grid(self.grid)

    def mv_scatter(self, Xs, y, title=None):
        ori_dims = torch.tensor([X.shape[1] for X in Xs])
        if self.manifold is not None and self.manifold.n_components < torch.max(ori_dims):
            Xs = self.__embed__(Xs, y)
        dims = torch.tensor([X.shape[1] for X in Xs])
        max_dim = torch.max(dims)
        Xs = DataVisualizer.__group__(Xs, y)
        assert 0 < max_dim <= 3
        if not self.pausing:
            if max_dim <= 2:
                self.fig, self.ax = plt.subplots()
                if max_dim == 1:
                    max_dim = 2
            elif max_dim == 3:
                self.fig = plt.figure()
                self.ax = Axes3D(self.fig)
            plt.rc('grid', linestyle="dotted", color='black', alpha=0.6)
        else:
            self.ax.clear()
        if title is not None:
            self.fig.canvas.set_window_title(title)

        for v in range(len(Xs)):
            labels = y if y is not None else np.arange(len(Xs[v]))
            if dims[v] < max_dim:
                Xs[v] = [np.array([np.concatenate([x, np.zeros(max_dim - dims[v])], axis=0) for x in X_i]) for X_i in
                         Xs[v]]
            if max_dim <= 2:
                for i, X_i in enumerate(Xs[v]):
                    self.ax.scatter(X_i[:, 0], X_i[:, 1],
                                    c=self.cmaps[i % len(self.cmaps)],
                                    marker=self.markers[v % len(self.markers)],
                                    s=50, label=labels[i], **self.scatter_params)
            elif max_dim == 3:
                for i, X_i in enumerate(Xs[v]):
                    self.ax.scatter(X_i[:, 0], X_i[:, 1], X_i[:, 2],
                                    c=self.cmaps[i % len(self.cmaps)],
                                    marker=self.markers[v % len(self.markers)],
                                    s=50, label=labels[i], **self.scatter_params)
            else:
                raise AttributeError('Unable to plot space of dimension greater than 3!')
        if self.legend: self.ax.legend()

        if (ori_dims == dims).all():
            self.ax.set_title('{}D feature space'.format(dims.tolist()))
        else:
            self.ax.set_title('{}D embeddings of {}D feature space'.format(dims.tolist(), ori_dims.tolist()))
        plt.grid(self.grid)

    def pause(self, interval=0.001):
        self.pausing = True
        plt.pause(interval)

    def show(self, block=True):
        self.pausing = False
        plt.show(block=block)

    def save(self, file):
        plt.savefig(file)

    @staticmethod
    def __group__(Xs, y):
        y_unique = np.unique(y)
        Rs = []
        for X in Xs:
            Rs.append([X[np.where(y == c)[0]] for c in y_unique])
        return Rs

    def __embed__(self, Xs, y):
        if self.manifold_style == 'per_view':
            return [self.manifold.fit_transform(X, y) for X in Xs]
        elif self.manifold_style == 'global':
            n_views = len(Xs)
            X_globe = torch.cat(Xs)
            y_globe = torch.cat([y for _ in range(n_views)])
            X_mbed = self.manifold.fit_transform(X_globe, y_globe)
            Xs = [X_mbed[int(len(X_mbed) / n_views * _):int(len(X_mbed) / n_views * (_ + 1))] for _ in range(n_views)]
            return Xs
