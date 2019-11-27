from .typing import *
from . import pre_vectorize, pre_listify
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import numpy as np
import itertools
import torch


class DataVisualizer:
    cmaps = ['red', 'green', 'blue', 'yellow', 'cyan', 'orange', 'purple', 'teal', 'steelblue', 'crimson', 'pink',
             'navy', 'grey']
    markers = ['o', '^', 's', '*', 'p', 'P', 'v', 'X', 'D', 'H', "2", '$...$']

    figure_params = {'figsize': (4, 4)}
    scatter_params = {'linewidth': 0.1, 'alpha': 0.5}
    legend_params = {'fancybox': True, 'prop': {'size': 8}}

    axe3d_scale = 1.22
    axe3d_title_position = 1.08

    def __init__(self,
                 embed_algo=None,
                 embed_style='global',
                 force_embed=False,
                 grid=True,
                 legend=True):
        assert embed_algo is None or embed_style in ['per_view', 'global']
        self.manifold = None
        if isinstance(embed_algo, str):
            raise UserWarning('Not supporting String aliases for dimensionality reduction algorithms.')
        elif hasattr(embed_algo, 'n_components'):
            self.manifold = embed_algo
        elif callable(embed_algo):
            self.manifold = embed_algo()
            assert hasattr(self.manifold, 'n_components'), 'Unrecognized dimensionality reduction algorithm.'
        self.manifold_style = embed_style
        self.force_embed: bool = force_embed
        self.pausing: bool = False
        self.grid: bool = grid
        self.legend: bool = legend
        self.fig = None
        self.axes = []

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

    @pre_vectorize(positionals=(1, 2))
    def scatter(self,
                X,
                y=None,
                dim=2,
                title=None):
        n_samples = X.shape[0]
        ori_dim = X.shape[1]
        y = np.array([0 for _ in range(n_samples)]) if y is None else y
        y_unique = np.unique(y)
        X = self.__embed(X, y, dim)
        dim = X.shape[1]
        X = self.__group(X, y)

        ax = self.__init_axe(dim=dim)

        if dim <= 2:
            for i, X_i in enumerate(X):
                ax.scatter(X_i[:, 0], X_i[:, 1] if dim == 2 else np.zeros(X_i.shape[0]),
                           c=self.cmaps[i % len(self.cmaps)] if len(y_unique) > 1 else self.cmaps[-1],
                           s=50,
                           **self.scatter_params)
        elif dim == 3:
            for i, X_i in enumerate(X):
                ax.scatter(X_i[:, 0], X_i[:, 1], X_i[:, 2],
                           c=self.cmaps[i % len(self.cmaps)] if len(y_unique) > 1 else self.cmaps[-1],
                           s=50,
                           **self.scatter_params)
        else:
            raise AttributeError('Unable to plot space of dimension greater than 3!')
        if self.legend:
            ax.add_artist(self.__cls_legend(y_unique))

        if title is not None:
            ax.set_title(title, y=self.axe3d_title_position if dim == 3 else 1)
        elif ori_dim == dim:
            ax.set_title('{}D feature space'.format(dim), y=self.axe3d_title_position if dim == 3 else 1)
        else:
            ax.set_title('{}D embeddings of {}D feature space'.format(dim, ori_dim),
                         y=self.axe3d_title_position if dim == 3 else 1)
        self.axes.append(ax)

    @pre_vectorize(positionals=(1, 2))
    def mv_scatter(self,
                   Xs,
                   y=None,
                   dim=2,
                   title=None):
        n_views = Xs.shape[0]
        n_samples = Xs.shape[1]
        y = np.array([0 for _ in range(n_samples)]) if y is None else y
        y_unique = np.unique(y)
        ori_dims = np.array([X.shape[1] for X in Xs])
        Xs = self.__embed(Xs, y, dim)
        dims = np.array([X.shape[1] for X in Xs])
        max_dim = np.max(dims)
        Xs = self.__group(Xs, y)

        ax = self.__init_axe(dim=max_dim)

        for v in range(n_views):
            if dims[v] < max_dim:
                Xs[v] = [np.array([np.concatenate([x, np.zeros(max_dim - dims[v])], axis=0) for x in X_i]) for X_i in
                         Xs[v]]
            if max_dim <= 2:
                for i, X_i in enumerate(Xs[v]):
                    ax.scatter(X_i[:, 0], X_i[:, 1] if dims[v] == 2 else np.zeros(X_i.shape[0]),
                               c=self.cmaps[i % len(self.cmaps)] if len(y_unique) > 1 else self.cmaps[-1],
                               marker=self.markers[v % len(self.markers)],
                               s=50,
                               **self.scatter_params)
            elif max_dim == 3:
                for i, X_i in enumerate(Xs[v]):
                    ax.scatter(X_i[:, 0], X_i[:, 1], X_i[:, 2],
                               c=self.cmaps[i % len(self.cmaps)] if len(y_unique) > 1 else self.cmaps[-1],
                               marker=self.markers[v % len(self.markers)],
                               s=50,
                               **self.scatter_params)
            else:
                raise AttributeError('Unable to plot space of dimension greater than 3!')
        if self.legend:
            ax.add_artist(self.__cls_legend(y_unique))
            ax.add_artist(self.__view_legend(n_views))

        if title is not None:
            ax.set_title(title, y=self.axe3d_title_position if max_dim == 3 else 1)
        elif (ori_dims == dims).all():
            ax.set_title('{}D feature space'.format(dims.tolist()), y=self.axe3d_title_position if max_dim == 3 else 1)
        else:
            ax.set_title('{}D embeddings of {}D feature space'.format(dims.tolist(), ori_dims.tolist()),
                         y=self.axe3d_title_position if max_dim == 3 else 1)
        self.axes.append(ax)

    def pause(self, interval: Number = 0.001):
        """
        Pause for rerender figure.

        :param interval:
        :return:
        """
        self.pausing = True
        plt.pause(interval)

    def show(self,
             grids: Optional[Union[String, Sequence[Tuple[Integer, Integer, Union[Integer, slice]]]]] = 'auto',
             title: Optional[String] = None,
             block: Boolean = True):
        """
        Show figure in window.

        :param grids:
        :param title:
        :param block:
        :return:
        """
        if title is not None:
            self.fig.canvas.set_window_title(title)
        fig_size = self.fig.get_size_inches()
        self.fig.set_size_inches(fig_size[0] * len(self.axes), fig_size[1])
        self.__rearrange_axes(grids)
        plt.show(block=block)
        self.__clear_fig()

    def save(self,
             file: String):
        """
        Save figure to file.

        :param file:
        :return:
        """
        self.fig.savefig(file)

    def __init_axe(self, ax=None, dim=2):
        if self.fig is None:
            self.fig = plt.figure(**self.figure_params)
        if not self.pausing:
            new_plot_pos = (1, len(self.axes) + 1, 1)
            if dim <= 2:
                ax = self.fig.add_subplot(*new_plot_pos)
            elif dim == 3:
                ax = self.fig.add_subplot(*new_plot_pos, projection='3d')
                ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([self.axe3d_scale] * 3 + [1]))
            plt.rc('grid', linestyle="dotted", color='black', alpha=0.6)
        else:
            ax.clear()
        ax.grid(self.grid)
        return ax

    def __rearrange_axes(self,
                         grids: Optional[Union[String, Sequence[Tuple[Integer, Integer, Union[Integer, slice]]]]] = 'auto'):
        if grids is None or grids == 'auto':
            gs = gridspec.GridSpec(1, len(self.axes))
            for _ in range(len(self.axes)):
                self.axes[_].set_position(gs[_].get_position(self.fig))
                self.axes[_].set_subplotspec(gs[_])
        else:
            for _, grid in enumerate(grids):
                grid = list(grid)
                gs = gridspec.GridSpec(*grid[:2])
                self.axes[_].set_position(gs[grid[2]].get_position(self.fig))
                self.axes[_].set_subplotspec(gs[grid[2]])

    def __clear_fig(self):
        self.pausing = False
        self.fig = None
        self.axes.clear()
        plt.close(self.fig)

    @pre_listify(positionals=1)
    def __cls_legend(self, classes):
        if len(classes) > 1:
            handles = [patches.Patch(color=self.cmaps[_ % len(self.cmaps)], label=cls)
                       for _, cls in enumerate(classes)]
        else:
            handles = [patches.Patch(color=self.cmaps[-1], label=classes[0])]
        legend = plt.legend(handles=handles, title='Classes', loc=1, **self.legend_params)
        plt.setp(legend.get_title(), fontsize=8)
        return legend

    def __view_legend(self, n_views):
        handles = [lines.Line2D([0], [0],
                                label='${}^{{{}}}$'.format(_,
                                                           'st' if _ % 10 == 1 else 'nd' if _ % 10 == 2
                                                           else 'rd' if _ % 10 == 3 else 'th'),
                                marker=self.markers[(_ - 1) % len(self.markers)],
                                markerfacecolor='w',
                                markersize=8,
                                color='black',
                                fillstyle='none',
                                linestyle='None',
                                alpha=0.5)
                   for _ in range(1, n_views + 1)]
        legend = plt.legend(handles=handles, title='Views', loc=4, **self.legend_params)
        plt.setp(legend.get_title(), fontsize=8)
        return legend

    @staticmethod
    def __group(Xs, y, y_unique=None):
        y_unique = np.unique(y) if y_unique is None else y_unique
        if len(Xs.shape) == 2:
            return [Xs[np.where(y == c)[0]] for c in y_unique]
        else:
            Rs = []
            for X in Xs:
                Rs.append([X[np.where(y == c)[0]] for c in y_unique])
            return Rs

    def __embed(self, Xs, y, dim=2):
        if self.manifold is None or (Xs.shape[-1] <= 3 and not self.force_embed):
            return Xs
        self.manifold.n_components = dim
        if len(Xs.shape) == 2:
            return self.manifold.fit_transform(Xs, y)
        elif self.manifold_style == 'per_view':
            return [self.manifold.fit_transform(X, y) for X in Xs]
        elif self.manifold_style == 'global':
            n_views = len(Xs)
            X_globe = torch.cat(Xs)
            y_globe = torch.cat([y for _ in range(n_views)])
            X_mbed = self.manifold.fit_transform(X_globe, y_globe)
            Xs = [X_mbed[int(len(X_mbed) / n_views * _):int(len(X_mbed) / n_views * (_ + 1))] for _ in range(n_views)]
            return Xs
