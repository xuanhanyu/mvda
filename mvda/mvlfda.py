from .bases import MvDAlgoBase
from .objectives import MvLFDAIntraScatter, MvLFDAInterScatter, ViewConsistency, Regularization


class MvLFDA(MvDAlgoBase):
    name = 'Multi-view Discriminant Analysis'

    def __init__(self,
                 n_components='auto',
                 ep='eig',
                 reg='auto',
                 kernels=None,
                 affinity_type='kernel',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1):
        super(MvLFDA, self).__init__(n_components=n_components,
                                     ep=ep,
                                     reg=reg,
                                     kernels=kernels)
        self.swo = MvLFDAIntraScatter(affinity_type, n_neighbors, epsilon, affinity_kernel, gamma)
        self.sbo = MvLFDAInterScatter(affinity_type, n_neighbors, epsilon, affinity_kernel, gamma)

    def _Sw_(self):
        return self.swo.target()

    def _Sb_(self):
        return self.sbo.target()


class MvLFDAvc(MvLFDA):
    name = MvLFDA.name + ' with View-Consistency'

    def __init__(self,
                 n_components='auto',
                 ep='eig',
                 reg='auto',
                 kernels=None,
                 affinity_type='kernel',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1,
                 lambda_vc=0.01):
        super(MvLFDAvc, self).__init__(n_components=n_components,
                                       ep=ep,
                                       reg=reg,
                                       kernels=kernels,
                                       affinity_type=affinity_type,
                                       n_neighbors=n_neighbors,
                                       epsilon=epsilon,
                                       affinity_kernel=affinity_kernel,
                                       gamma=gamma)
        self.lambda_vc = lambda_vc
        self.vco = ViewConsistency(reg=reg)

    def calculate_objectives(self):
        self.Sw = self._Sw_() + self.lambda_vc * self.vco.target()
        self.Sb = self._Sb_()


class RMvLFDA(MvLFDA):
    name = 'Regularized ' + MvLFDA.name

    def __init__(self,
                 n_components='auto',
                 ep='eig',
                 reg='auto',
                 kernels=None,
                 affinity_type='kernel',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1,
                 lambda_reg=0.1):
        super(RMvLFDA, self).__init__(n_components=n_components,
                                      ep=ep,
                                      reg=reg,
                                      kernels=kernels,
                                      affinity_type=affinity_type,
                                      n_neighbors=n_neighbors,
                                      epsilon=epsilon,
                                      affinity_kernel=affinity_kernel,
                                      gamma=gamma)
        self.lambda_reg = lambda_reg
        self.ro = Regularization()

    def calculate_objectives(self):
        self.Sw = self._Sw_() + self.lambda_reg * self.ro.target()
        self.Sb = self._Sb_()


class RMvLFDAvc(MvLFDAvc):
    name = 'Regularized ' + MvLFDAvc.name

    def __init__(self,
                 n_components='auto',
                 ep='eig',
                 reg='auto',
                 kernels=None,
                 affinity_type='kernel',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1,
                 lambda_vc=0.01, lambda_reg=0.1):
        super(RMvLFDAvc, self).__init__(n_components=n_components,
                                        ep=ep,
                                        reg=reg,
                                        kernels=kernels,
                                        affinity_type=affinity_type,
                                        n_neighbors=n_neighbors,
                                        epsilon=epsilon,
                                        affinity_kernel=affinity_kernel,
                                        gamma=gamma,
                                        lambda_vc=lambda_vc)
        self.lambda_reg = lambda_reg
        self.ro = Regularization()

    def calculate_objectives(self):
        self.Sw = self._Sw_() + self.lambda_vc * self.vco.target() + self.lambda_reg * self.ro.target()
        self.Sb = self._Sb_()
