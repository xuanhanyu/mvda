from epsolver import EPSolver
import torch
import numpy as np


def class_vectors(y):
    y_unique = torch.unique(torch.tensor(y))
    return [torch.tensor([1 if _ == clazz else 0 for _ in y]) for clazz in y_unique]


def within_class_vars(mv_Xs, y, lambda_vc=0.0):
    num_views = len(mv_Xs)
    y_unique = torch.unique(torch.tensor(y))
    ecs = [torch.tensor([1 if _ == ci else 0 for _ in y], dtype=torch.float) for ci in y_unique]

    W = torch.zeros(len(y), len(y))
    for ci in y_unique:
        W += ecs[ci].unsqueeze(0).t() @ ecs[ci].unsqueeze(0) / (torch.sum(ecs[ci]) * len(mv_Xs))
    D = torch.eye(len(y))
    Ireg_vc = torch.eye(len(y)) * 1e-5

    S_cols = []
    for j in range(num_views):
        # Ireg_j = torch.eye(len(y)) * torch.trace(mv_Xs[j] @ mv_Xs[j].t()) * 0.0001
        S_rows = []
        for r in range(num_views):
            # Ireg_r = torch.eye(len(y)) * torch.trace(mv_Xs[r] @ mv_Xs[r].t()) * 0.0001
            if j == r:
                s_jr = D - W
                vc_jr = 2 * num_views * (mv_Xs[j] @ mv_Xs[j].t() @ mv_Xs[j] @ mv_Xs[j].t() + Ireg_vc).inverse() - \
                    2 * (mv_Xs[r] @ mv_Xs[r].t() @ mv_Xs[j] @ mv_Xs[j].t() + Ireg_vc).inverse()
            else:
                s_jr = -W
                vc_jr = -2 * (mv_Xs[r] @ mv_Xs[r].t() @ mv_Xs[j] @ mv_Xs[j].t() + Ireg_vc).inverse()

            s_jr = mv_Xs[j].t() @ (s_jr + lambda_vc * vc_jr) @ mv_Xs[r]
            S_rows.append(s_jr)
        S_cols.append(torch.cat(S_rows, dim=1))
    Sw = torch.cat(S_cols, dim=0)
    return Sw


def between_class_vars(mv_Xs, y):
    num_views = len(mv_Xs)
    y_unique = torch.unique(torch.tensor(y))
    ecs = [torch.tensor([1 if _ == ci else 0 for _ in y], dtype=torch.float) for ci in y_unique]

    n = len(mv_Xs) * mv_Xs[0].shape[0]
    W = torch.zeros(len(y), len(y))
    for ci in y_unique:
        W += ecs[ci].unsqueeze(0).t() @ ecs[ci].unsqueeze(0) / (torch.sum(ecs[ci]) * len(mv_Xs))
    B = torch.ones(len(y), len(y)) / n

    S_cols = []
    for j in range(num_views):
        S_rows = []
        for r in range(num_views):
            s_jr = mv_Xs[j].t() @ (W - B) @ mv_Xs[r]
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


if __name__ == '__main__':
    use_kernel = False

    def main():
        import synthetics
        mv_Xs, y = synthetics.single_blob_dataset()

        # kernelize
        from sklearn.metrics.pairwise import rbf_kernel as kernel
        mv_Ks = [torch.tensor(kernel(mv_Xs[_])).float() if use_kernel else mv_Xs[_] for _ in range(len(mv_Xs))]
        dims = [Ks.shape[1] for Ks in mv_Ks]

        Sw = within_class_vars(mv_Ks, y, lambda_vc=0.1)
        Sb = between_class_vars(mv_Ks, y)

        solver = EPSolver(algo='eig')
        eigen_vecs = solver.solve(Sw, Sb)
        Ws = projections(eigen_vecs, dims)
        print('Projection matrices:', [W.shape for W in Ws])

        # transform
        mv_Ys = [(Ws[_].t() @ mv_Ks[_].t()).t() for _ in range(len(mv_Ks))]
        mv_Ys = group(mv_Ys, y)
        mv_Ys = [[Y[:, :2] for Y in Ys] for Ys in mv_Ys]

        # plot
        from data_visualizer import DataVisualizer
        dv = DataVisualizer()
        dv.mv_scatter(group(mv_Xs, y))
        dv.mv_scatter(mv_Ys)
        dv.show()


    main()
