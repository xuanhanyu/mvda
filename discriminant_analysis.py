import itertools
import torch
import numpy as np


# -------------------------------------------------
# LINEAR DISCRIMINANT ANALYSIS
# -------------------------------------------------
def class_means(ws):
    """
    Compute class means u(l) of each class l
    """
    us = [torch.mean(w, dim=0) for w in ws]
    u = mean(ws)
    return us, u


def mean(ws):
    return torch.mean(torch.cat(ws, dim=0), dim=0)


def within_class_vars(ws, us=None):
    """
    Compute within class variance matrices SW(l) of each class l
    """
    if us is None:
        us = [torch.mean(w, dim=0) for w in ws]
    SWs = [torch.mm((w_l - u_l).t(), (w_l - u_l)) for w_l, u_l in zip(ws, us)]
    SW = torch.sum(torch.cat([SW_l.unsqueeze(0) for SW_l in SWs], dim=0), dim=0)
    return SWs, SW


def between_class_vars(ws, us=None, u=None):
    """
    Compute between class variance matrices SB(l) of each class l
    """
    if us is None:
        us = [torch.mean(w, dim=0) for w in ws]
    if u is None:
        u = mean(ws)
    SBs = [len(w_l) * torch.mm((u_l - u).unsqueeze(0).t(), (u_l - u).unsqueeze(0)) for w_l, u_l in zip(ws, us)]
    SB = torch.sum(torch.cat([SB_l.unsqueeze(0) for SB_l in SBs], dim=0), dim=0)
    return SBs, SB


def local_between_class_vars(ws, us):
    """
    Compute local between class variance matrices SlB(ij) of each pair of classes i and j
    Return SlB = sum(SlBij)
    """
    indexes = list(range(len(ws)))
    pairs = list(itertools.permutations(indexes, 2))
    SlBs = [torch.mm((us[i] - us[j]).unsqueeze(0).t(), (us[i] - us[j]).unsqueeze(0)) for i, j in pairs]
    SlB = torch.sum(torch.cat([SlB_l.unsqueeze(0) for SlB_l in SlBs], dim=0), dim=0)
    return SlB


def eigen(W, argmin=False):
    """
    Compute global projection vectors V (class independent method)
    """
    eigen_vals, eigen_vecs = torch.eig(W, eigenvectors=True)
    eigen_pairs = [[eigen_vals[i][0], eigen_vecs.t()[i]] for i in range(eigen_vals.shape[0])]
    eigen_pairs = sorted(eigen_pairs, key=lambda ep: torch.abs(ep[0]).item(), reverse=not argmin)
    eigen_vecs = torch.cat([eigen_pair[1].unsqueeze(0) for eigen_pair in eigen_pairs], dim=0).t()
    eigen_vals = torch.cat([eigen_pair[0].unsqueeze(0) for eigen_pair in eigen_pairs], dim=0)
    return eigen_vals, eigen_vecs


def projection(eigen_vecs, n_components=None):
    if n_components is None:
        n_components = eigen_vecs.shape[1]
    return eigen_vecs[:, :n_components]


def class_transformations(SWs, SB):
    """
    Compute transformations W(l) of each class l (class-dependent method)
    """
    return [torch.mm(SW_l.inverse(), SB) for SW_l in SWs]


def class_projections(Ws, n_component):
    """
    Compute projection vectors V(l) of each class l (class-dependent method)
    """
    Vs = []
    for W_l in Ws:
        eigen_vals, eigen_vecs = torch.eig(W_l, eigenvectors=True)
        eigen_pairs = [[torch.abs(eigen_vals[i][0]), eigen_vecs.t()[i]] for i in range(eigen_vals.shape[0])]
        eigen_pairs = sorted(eigen_pairs, key=lambda ep: ep[0].item(), reverse=True)
        Vs.append(torch.cat([eigen_pairs[_][1].unsqueeze(0) for _ in range(n_component)], dim=0).t())
    return Vs


# -------------------------------------------------
# KERNEL FISHER DISCRIMINANT ANALYSIS
# -------------------------------------------------
def kfda(Ks, lmb=0.001):
    ns = [K_i.shape[1] for K_i in Ks]
    n = sum(ns)

    Ns = [torch.mm(torch.mm(K_i, torch.eye(n_i) - 1 / n_i), K_i.t()) for K_i, n_i in zip(Ks, ns)]
    N = torch.sum(torch.cat([N_i.unsqueeze(0) for N_i in Ns], dim=0), dim=0) + torch.diag(torch.tensor(lmb).repeat(n))

    Ms = [torch.sum(K_i, dim=1) / float(n_i) for K_i, n_i in zip(Ks, ns)]
    M_asterick = torch.sum(torch.cat([K_i for K_i in Ks], dim=1), dim=0) / float(n)
    M = torch.sum(torch.cat([M_i.unsqueeze(0) for M_i in
                            [n_i * (M_i - M_asterick).unsqueeze(0) * (M_i - M_asterick).unsqueeze(0).t()
                             for n_i, M_i in zip(ns, Ms)]], dim=0), dim=0)

    eigen_vals, eigen_vecs = eigen(N.inverse() @ M)
    A = projection(eigen_vecs, 2)
    return A


# -------------------------------------------------
# ANGULAR DISCRIMINANT ANALYSIS
# -------------------------------------------------
def within_class_angular_vars(ws, us=None):
    """
    Compute within class angular variance matrices SW(l) of each class l
    """
    if us is None:
        us = [torch.mean(w, dim=0) for w in ws]
    OWs = [torch.sum(torch.cat([(u_l.unsqueeze(0).t() @ x_i.unsqueeze(0)).unsqueeze(0) for x_i in w_l], dim=0),
                     dim=0) for w_l, u_l in zip(ws, us)]
    OW = torch.sum(torch.cat([OW_l.unsqueeze(0) for OW_l in OWs], dim=0), dim=0)
    return OW


def wcav(X, y):
    y_unique = torch.unique(torch.tensor(y))
    ecs = [torch.tensor([1 if _ == ci else 0 for _ in y], dtype=torch.float) for ci in y_unique]
    Ow = torch.zeros(len(y), len(y))
    for ci in y_unique:
        Ow += ecs[ci].unsqueeze(0).t() @ ecs[ci].unsqueeze(0) / torch.sum(ecs[ci])
    return X.t() @ Ow @ X


def bcav(X, y):
    y_unique = torch.unique(torch.tensor(y))
    ecs = [torch.tensor([1 if _ == ci else 0 for _ in y], dtype=torch.float) for ci in y_unique]
    Ob = torch.zeros((len(y), len(y)))
    for ca in y_unique:
        for cb in y_unique:
            Ob += ecs[ca].unsqueeze(0).t() @ ecs[cb].unsqueeze(0) / len(X)
    return X.t() @ Ob @ X


def between_class_angular_vars(ws, us=None, u=None):
    """
    Compute between class angular variance matrices SW(l) of each class l
    """
    if us is None:
        us = [torch.mean(w, dim=0) for w in ws]
    if u is None:
        u = mean(ws)
    OBs = [len(w_l) * (u.unsqueeze(0).t() @ u_l.unsqueeze(0)) for w_l, u_l in zip(ws, us)]
    OB = torch.sum(torch.cat([OB_l.unsqueeze(0) for OB_l in OBs], dim=0), dim=0)
    return OB

