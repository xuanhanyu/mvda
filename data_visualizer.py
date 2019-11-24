from torchsl.utils import pre_tensorize
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch


class DataVisualizer:
    cmaps = ['red', 'green', 'blue', 'yellow', 'cyan', 'orange', 'purple', 'teal', 'steelblue', 'crimson', 'pink',
             'navy']
    markers = ['o', '^', 's', '*', 'p', 'P', 'v', 'X', 'D', 'H', "2", '$...$']
    scatter_params = {'linewidth': 0.1, 'alpha': 0.5}

    def __init__(self, embed_algo=None, embed_style='global', grid=True, legend=True):
        if embed_algo is not None and (hasattr(embed_algo, 'n_components') and embed_algo.n_components > 3):
            raise AssertionError
        assert embed_algo is None or embed_style in ['per_view', 'global']
        self.manifold = embed_algo
        self.manifold_style = embed_style
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

    @pre_tensorize(positionals=(1, 2))
    def scatter(self, X, y, title=None):
        ori_dim = X.shape[1]
        if self.manifold is not None and self.manifold.n_components < ori_dim:
            X = self.__embed(X, y)
        plt.rc('grid', linestyle="dotted", color='black', alpha=0.6)
        dim = X.shape[1]
        X = DataVisualizer.__group__(X, y)
        assert 0 < dim <= 3
        if not self.pausing:
            if dim <= 2:
                self.fig, self.ax = plt.subplots()
                if dim == 1:
                    dim = 2
            elif dim == 3:
                self.fig = plt.figure()
                self.ax = Axes3D(self.fig)
            plt.rc('grid', linestyle="dotted", color='black', alpha=0.6)
        else:
            self.ax.clear()
        if title is not None:
            self.fig.canvas.set_window_title(title)

        y_unique = torch.unique(y)  # y if y is not None else np.arange(len(X))
        if dim <= 2:
            for i, X_i in enumerate(X):
                self.ax.scatter(X_i[:, 0], X_i[:, 1],
                                c=self.cmaps[i % len(self.cmaps)],
                                # marker=self.markers[v % len(self.markers)],
                                s=50, label=y_unique[i], **self.scatter_params)
        elif dim == 3:
            for i, X_i in enumerate(X):
                self.ax.scatter(X_i[:, 0], X_i[:, 1], X_i[:, 2],
                                c=self.cmaps[i % len(self.cmaps)],
                                # marker=self.markers[v % len(self.markers)],
                                s=50, label=y_unique[i], **self.scatter_params)
        else:
            raise AttributeError('Unable to plot space of dimension greater than 3!')
        if self.legend: self.ax.legend()

        if ori_dim == dim:
            self.ax.set_title('{}D feature space'.format(dim))
        else:
            self.ax.set_title('{}D embeddings of {}D feature space'.format(dim, ori_dim))
        plt.grid(self.grid)

    @pre_tensorize(positionals=(1, 2))
    def mv_scatter(self, Xs, y, title=None):
        ori_dims = torch.tensor([X.shape[1] for X in Xs])
        if self.manifold is not None and self.manifold.n_components < torch.max(ori_dims):
            Xs = self.__mv_embed(Xs, y)
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

        y_unique = torch.unique(y)
        for v in range(len(Xs)):
            if dims[v] < max_dim:
                Xs[v] = [np.array([np.concatenate([x, np.zeros(max_dim - dims[v])], axis=0) for x in X_i]) for X_i in
                         Xs[v]]
            if max_dim <= 2:
                for i, X_i in enumerate(Xs[v]):
                    self.ax.scatter(X_i[:, 0], X_i[:, 1],
                                    c=self.cmaps[i % len(self.cmaps)],
                                    marker=self.markers[v % len(self.markers)],
                                    s=50, label=y_unique[i], **self.scatter_params)
            elif max_dim == 3:
                for i, X_i in enumerate(Xs[v]):
                    self.ax.scatter(X_i[:, 0], X_i[:, 1], X_i[:, 2],
                                    c=self.cmaps[i % len(self.cmaps)],
                                    marker=self.markers[v % len(self.markers)],
                                    s=50, label=y_unique[i], **self.scatter_params)
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
        if len(Xs.shape) == 2:
            return [Xs[np.where(y == c)[0]] for c in y_unique]
        else:
            Rs = []
            for X in Xs:
                Rs.append([X[np.where(y == c)[0]] for c in y_unique])
            return Rs

    def __embed(self, X, y):
        return self.manifold.fit_transform(X, y)

    def __mv_embed(self, Xs, y):
        if self.manifold_style == 'per_view':
            return [self.manifold.fit_transform(X, y) for X in Xs]
        elif self.manifold_style == 'global':
            n_views = len(Xs)
            X_globe = torch.cat(Xs)
            y_globe = torch.cat([y for _ in range(n_views)])
            X_mbed = self.manifold.fit_transform(X_globe, y_globe)
            Xs = [X_mbed[int(len(X_mbed) / n_views * _):int(len(X_mbed) / n_views * (_ + 1))] for _ in range(n_views)]
            return Xs
