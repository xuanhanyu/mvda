from epsolver import EPSolver
import torch
import numpy as np


def class_means(mv_Xs):
    sv_us = [[torch.mean(w, dim=0) for w in ws] for ws in mv_Xs]
    # mv_us = [torch.mean(torch.cat([ws[i] for ws in mv_ws])) for i in range(len(mv_ws[0]))]
    return sv_us


def mean(mv_Xs):
    return [torch.mean(torch.cat(ws), dim=0) for ws in mv_Xs], \
           torch.mean(torch.cat([torch.cat(ws) for ws in mv_Xs]), dim=0)


def num_samples(mv_Xs):
    return [[len(w_i) for w_i in ws] for ws in mv_Xs], [len([ws[i] for ws in mv_Xs]) for i in
                                                        range(len(mv_Xs[0]))]


def dimensions(mv_Xs):
    return [len(ws[0][0]) for ws in mv_Xs]


def within_class_vars(mv_Xs, sv_us, y):
    num_views = len(mv_Xs)
    dims = dimensions(mv_Xs)
    y_unique = torch.unique(torch.tensor(y))
    sv_ns, mv_ns = num_samples(mv_Xs)

    S_cols = []
    for j in range(num_views):
        S_rows = []
        for r in range(num_views):
            s_jr = torch.zeros((dims[j], dims[r]))
            v_jr = torch.zeros((dims[j], dims[r]))
            for i in range(len(y_unique)):
                s_jr -= sv_ns[j][i] * sv_ns[r][i] / mv_ns[i] * (
                        sv_us[j][i].unsqueeze(0).t() @ sv_us[r][i].unsqueeze(0))
                v_jr += mv_Xs[j][i].t() @ mv_Xs[j][i] if j == r else 0
            S_rows.append(s_jr + v_jr)
            if j == r == 1:
                print(-s_jr, mv_ns[0])
        S_cols.append(torch.cat(S_rows, dim=1))
    Sw = torch.cat(S_cols, dim=0)
    return Sw


def between_class_vars(mv_Xs, sv_us, y):
    num_views = len(mv_Xs)
    dims = dimensions(mv_Xs)
    y_unique = torch.unique(torch.tensor(y))
    sv_ns, mv_ns = num_samples(mv_Xs)
    n = sum(mv_ns)

    S_cols = []
    for j in range(num_views):
        mean_j = torch.sum(torch.cat([sv_ns[j][i] * sv_us[j][i].unsqueeze(0) for i in range(len(y_unique))]), dim=0)
        S_rows = []
        for r in range(num_views):
            mean_r = torch.sum(torch.cat([sv_ns[r][i] * sv_us[r][i].unsqueeze(0) for i in range(len(y_unique))]), dim=0)

            d_jr = torch.zeros((dims[j], dims[r]))
            for i in range(len(y_unique)):
                d_jr += sv_ns[j][i] * sv_ns[r][i] / mv_ns[i] * (sv_us[j][i].unsqueeze(0).t() @ sv_us[r][i].unsqueeze(0))
            q_jr = mean_j.unsqueeze(0).t() @ mean_r.unsqueeze(0)

            # g_jr = torch.zeros((dims[j], dims[r]))
            # h_jr = torch.zeros((dims[j], dims[r]))
            # for a in range(len(y_unique)):
            #     for b in range(len(y_unique)):
            #         g_jr += sv_ns[j][a] * sv_ns[r][a] / mv_ns[a] * (sv_us[j][a].unsqueeze(0).t() @ sv_us[r][a].unsqueeze(0))
            #         h_jr += sv_ns[j][a] * sv_ns[r][b] / mv_ns[b] * (sv_us[j][a].unsqueeze(0).t() @ sv_us[r][b].unsqueeze(0))

            s_ij = d_jr - q_jr / n
            # s_ij = 2 * g_jr - 2 * h_jr
            S_rows.append(s_ij)
            # print(d_jr - q_jr / n)
            # print(2 * g_jr  - 2 * h_jr)
        S_cols.append(torch.cat(S_rows, dim=1))
    Sb = torch.cat(S_cols, dim=0)
    return Sb * num_views


def projections(eigen_vecs, dims):
    return [eigen_vecs[sum(dims[:i]):sum(dims[:i+1]), ...] for i in range(len(dims))]


def group(mv_Ds, y):
    y_unique = np.unique(y)
    mv_Rs = []
    for Ds in mv_Ds:
        mv_Rs.append([Ds[np.where(y == c)[0]] for c in y_unique])
    return mv_Rs


if __name__ == '__main__':
    def main():
        import synthetics
        mv_Xs, y = synthetics.dual_blobs_dataset()
        mv_Ks = group(mv_Xs, y)
        # from scipy.io import loadmat
        # s = loadmat('cohai.mat')
        # mv_Xs = [torch.tensor(s['X'][0]), torch.tensor(s['X'][1])]
        # y = s['y'][0]
        # mv_Ks = group(mv_Xs, y)

        dims = dimensions(mv_Ks)
        sv_us = class_means(mv_Ks)
        Sw = within_class_vars(mv_Ks, sv_us, y)
        Sb = between_class_vars(mv_Ks, sv_us, y)

        solver = EPSolver(algo='eig')
        eigen_vecs = solver.solve(Sw, Sb)
        Ws = projections(eigen_vecs, dims)
        print('Projection matrices:', [W.shape for W in Ws])

        mv_Ys = [(Ws[_].t() @ mv_Xs[_].t()).t() for _ in range(len(mv_Xs))]
        mv_Ys = group(mv_Ys, y)
        mv_Ys = [[Y[:, :2] for Y in Ys] for Ys in mv_Ys]

        # plot
        from data_visualizer import DataVisualizer
        dv = DataVisualizer()
        dv.mv_scatter(mv_Ks, y)
        dv.mv_scatter(mv_Ys, y)
        dv.show()

    main()
