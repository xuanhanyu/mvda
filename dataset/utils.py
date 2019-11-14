import torch


def join_multiview_datasets(list_Xs, list_y=None):
    Xs_joined = []
    for _ in range(len(list_Xs[0])):
        tmp = []
        for Xs in list_Xs:
            tmp.append(Xs[_] if torch.is_tensor(Xs[_]) else torch.from_numpy(Xs[_]))
        Xs_joined.append(torch.cat(tmp))
    if list_y is None:
        return Xs_joined
    y_joined = torch.cat([y.unsqueeze(0) if torch.is_tensor(y) else torch.from_numpy(y).unsqueeze(0)
                          for y in list_y], dim=1).squeeze(0)
    return Xs_joined, y_joined


def multiview_train_test_split(Xs, y, p=0.8):
    n_samples = len(y)
    n_samples_train = int(n_samples * p)
    y_train, y_test = y[:n_samples_train], y[n_samples_train:]
    Xs_train, Xs_test = [X[:n_samples_train] for X in Xs], [X[n_samples_train:] for X in Xs]
    return Xs_train, y_train, Xs_test, y_test
