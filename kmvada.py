from epsolver import EPSolver
import torch
import numpy as np


def class_vectors(y):
    y_unique = torch.unique(torch.tensor(y))
    return [torch.tensor([1 if _ == clazz else 0 for _ in y]) for clazz in y_unique]


def within_class_vars(mv_Xs, y):
    num_views = len(mv_Xs)
    y_unique = torch.unique(torch.tensor(y))
    ecs = [torch.tensor([1 if _ == ci else 0 for _ in y], dtype=torch.float) for ci in y_unique]

    Ow = torch.zeros(len(y), len(y))
    for ci in y_unique:
        Ow += ecs[ci].unsqueeze(0).t() @ ecs[ci].unsqueeze(0) / (torch.sum(ecs[ci]) * num_views)

    S_cols = []
    for j in range(num_views):
        S_rows = []
        for r in range(num_views):
            s_jr = mv_Xs[j].t() @ Ow @ mv_Xs[r]
            S_rows.append(s_jr)
        S_cols.append(torch.cat(S_rows, dim=1))
    Sw = torch.cat(S_cols, dim=0)
    return Sw


def between_class_vars(mv_Xs, y):
    num_views = len(mv_Xs)
    y_unique = torch.unique(torch.tensor(y))
    ecs = [torch.tensor([1 if _ == ci else 0 for _ in y], dtype=torch.float) for ci in y_unique]

    n = len(mv_Xs) * len(y)
    Ob = torch.zeros(len(y), len(y))
    for ca in y_unique:
        for cb in y_unique:
            Ob += ecs[ca].unsqueeze(0).t() @ ecs[cb].unsqueeze(0) / n

    S_cols = []
    for j in range(num_views):
        S_rows = []
        for r in range(num_views):
            s_jr = mv_Xs[j].t() @ Ob @ mv_Xs[r]
            S_rows.append(s_jr)
        S_cols.append(torch.cat(S_rows, dim=1))
    Sb = torch.cat(S_cols, dim=0)
    return Sb


def projections(eigen_vecs, dims):
    return [eigen_vecs[sum(dims[:i]):sum(dims[:i + 1]), :] for i in range(len(dims))]


def group(mv_Ds, y):
    y_unique = np.unique(y)
    mv_Rs = []
    for Ds in mv_Ds:
        mv_Rs.append([Ds[np.where(y == c)[0]] for c in y_unique])
    return mv_Rs


def normalize(v):
    if isinstance(v, list):
        return [normalize(v_i) for v_i in v]
    elif len(v.shape) == 1:
        return torch.div(v, torch.norm(v, dim=0, p=2))
    return torch.cat([normalize(v_i).unsqueeze(0) for v_i in v], dim=0)


if __name__ == '__main__':
    use_kernel = False

    def main():
        import synthetics
        mv_Xs, y = synthetics.single_blob_dataset()

        # kernelize
        from sklearn.metrics.pairwise import rbf_kernel as kernel
        mv_Xs = [torch.tensor(kernel(mv_Xs[_])).float() if use_kernel else mv_Xs[_] for _ in range(len(mv_Xs))]
        mv_Ks = [normalize(Ks) for Ks in mv_Xs]
        dims = [Ks.shape[1] for Ks in mv_Ks]

        Sw = within_class_vars(mv_Ks, y)
        Sb = between_class_vars(mv_Ks, y)

        solver = EPSolver(algo='eig')
        eigen_vecs = solver.solve(Sw, Sb, argmin=True)
        Ws = projections(eigen_vecs, dims)

        # transform
        mv_Ys = [(Ws[_].t() @ mv_Xs[_].t()).t() for _ in range(len(mv_Xs))]
        mv_Ys = group(mv_Ys, y)
        mv_Ys = [[Y[:, :2] for Y in Ys] for Ys in mv_Ys]
        mv_Ys = [normalize(Ys) for Ys in mv_Ys]

        # plot
        from data_visualizer import DataVisualizer
        dv = DataVisualizer()
        dv.mv_scatter(group(mv_Ks, y))
        dv.mv_scatter(mv_Ys)
        dv.show()

    main()
