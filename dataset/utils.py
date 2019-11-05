import torch


def join_multiview_datasets(list_Xs, list_y=None):
    Xs_joined = []
    for _ in range(len(list_Xs[0])):
        tmp = []
        for Xs in list_Xs:
            tmp.append(Xs[_])
        Xs_joined.append(torch.cat(tmp))
    if list_y is None:
        return Xs_joined
    y_joined = torch.cat([y.unsqueeze(0) for y in list_y], dim=1).squeeze(0)
    return Xs_joined, y_joined
