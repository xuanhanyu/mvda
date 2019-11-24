from torchsl.mvsl.objectives import *
from torch.autograd import Variable
import torch
import torch.nn as nn


class SubNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SubNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.projection = nn.Linear(64, out_dim, bias=False)

    def forward(self, X):
        H = self.fc(X)
        Y = self.projection(H)
        if self.training:
            return Y, H, self.projection.weight
        return Y


class MvDNet(nn.Module):
    def __init__(self, in_dims, out_dim):
        super(MvDNet, self).__init__()
        self.subnets = nn.ModuleList([
            SubNet(in_dim, out_dim) for in_dim in in_dims
        ])

    @property
    def n_views(self):
        return len(self.subnets)

    def forward(self, Xs):
        Rs = [subnet.forward(X) for X, subnet in zip(Xs, self.subnets)]
        if self.training:
            Ys = [R[0] for R in Rs]
            Hs = [R[1] for R in Rs]
            Ws = [R[2] for R in Rs]
            return Ys, Hs, Ws
        return Rs


class MvDALoss(nn.Module):
    def __init__(self):
        super(MvDALoss, self).__init__()

    def forward(self, Hs, Ws, y):
        Sw = MvDAIntraScatter().fit(Hs, y).target()
        Sb = ClassSeparating().fit(Hs, y).target()
        W = torch.cat(Ws, dim=1)
        return torch.trace(W @ Sw @ W.t()) / torch.trace(W @ Sb @ W.t())


if __name__ == '__main__':

    def main():
        import synthetics
        from data_visualizer import DataVisualizer
        dv = DataVisualizer()

        mv_Xs, y = synthetics.random_dataset()

        dims = [Xs.shape[1] for Xs in mv_Xs]
        mvdnet = MvDNet(dims, 2)
        # mvdnet.eval()

        sample_Xs = [Variable(Xs) for Xs in mv_Xs]
        sample_y = y
        # print([H.shape for H in Hs])

        # plot
        dv.mv_scatter(mv_Xs, y)
        dv.show(False)

        criterion = MvDALoss()
        optim = torch.optim.Adam(mvdnet.parameters(), lr=1e-3)
        for i in range(200):
            optim.zero_grad()
            Ys, Hs, Ws = mvdnet(sample_Xs)
            loss = criterion(Hs, Ws, sample_y)
            loss.backward()
            optim.step()
            print('Loss:', loss)

            mvdnet.eval()
            with torch.no_grad():
                mv_Ys = mvdnet(mv_Xs)
                # mv_Ys = [[Y[:, :2] for Y in Ys] for Ys in mv_Ys]
                dv.mv_scatter(mv_Ys, y, title='{:03d}'.format(i + 1))
                dv.pause()

            mvdnet.train()
        dv.show()

    main()
