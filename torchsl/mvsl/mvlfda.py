from .bases import EOBasedMvSLAlgo
from .objectives import MvLFDAIntraScatter, MvLFDAInterScatter, ViewConsistency, Regularization


class MvLFDA(EOBasedMvSLAlgo):

    def __init__(self,
                 n_components='auto',
                 ep_algo='eigen',
                 ep_implementation='pytorch',
                 reg='auto',
                 kernels=None,
                 affinity_type='lle',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1,
                 lambda_lc=1.0,
                 *args, **kwargs):
        super(MvLFDA, self).__init__(n_components=n_components,
                                     ep_algo=ep_algo,
                                     ep_implementation=ep_implementation,
                                     reg=reg,
                                     kernels=kernels,
                                     *args, **kwargs)
        self.swo = MvLFDAIntraScatter(affinity_type=affinity_type,
                                      n_neighbors=n_neighbors,
                                      epsilon=epsilon,
                                      affinity_kernel=affinity_kernel,
                                      gamma=gamma,
                                      lambda_lc=lambda_lc)
        self.sbo = MvLFDAInterScatter(affinity_type=affinity_type,
                                      n_neighbors=n_neighbors,
                                      epsilon=epsilon,
                                      affinity_kernel=affinity_kernel,
                                      gamma=gamma,
                                      lambda_lc=lambda_lc)

    def _Sw_(self):
        return self.swo.target()

    def _Sb_(self):
        return self.sbo.target()


class MvLFDAvc(MvLFDA):

    def __init__(self,
                 n_components='auto',
                 ep_algo='eigen',
                 ep_implementation='pytorch',
                 reg='auto',
                 kernels=None,
                 affinity_type='kernel',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1,
                 lambda_lc=1.0, lambda_vc=0.01,
                 *args, **kwargs):
        super(MvLFDAvc, self).__init__(n_components=n_components,
                                       ep_algo=ep_algo,
                                       ep_implementation=ep_implementation,
                                       reg=reg,
                                       kernels=kernels,
                                       affinity_type=affinity_type,
                                       n_neighbors=n_neighbors,
                                       epsilon=epsilon,
                                       affinity_kernel=affinity_kernel,
                                       gamma=gamma,
                                       lambda_lc=lambda_lc,
                                       *args, **kwargs)
        self.lambda_vc = lambda_vc
        self.vco = ViewConsistency(reg=reg)

    def calculate_objectives(self):
        self.Sw = self._Sw_()
        self.Sw += self.lambda_vc * (self.Sw.trace() / self.vco.target().trace()) * self.vco.target()
        self.Sb = self._Sb_()


class RMvLFDA(MvLFDA):

    def __init__(self,
                 n_components='auto',
                 ep_algo='eigen',
                 ep_implementation='pytorch',
                 reg='auto',
                 kernels=None,
                 affinity_type='kernel',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1,
                 lambda_lc=1.0, lambda_reg=0.1,
                 *args, **kwargs):
        super(RMvLFDA, self).__init__(n_components=n_components,
                                      ep_algo=ep_algo,
                                      ep_implementation=ep_implementation,
                                      reg=reg,
                                      kernels=kernels,
                                      affinity_type=affinity_type,
                                      n_neighbors=n_neighbors,
                                      epsilon=epsilon,
                                      affinity_kernel=affinity_kernel,
                                      gamma=gamma,
                                      lambda_lc=lambda_lc,
                                      *args, **kwargs)
        self.lambda_reg = lambda_reg
        self.ro = Regularization()

    def calculate_objectives(self):
        self.Sw = self._Sw_() + self.lambda_reg * self.ro.target()
        self.Sb = self._Sb_()


class RMvLFDAvc(MvLFDAvc):

    def __init__(self,
                 n_components='auto',
                 ep_algo='eigen',
                 ep_implementation='pytorch',
                 reg='auto',
                 kernels=None,
                 affinity_type='kernel',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1,
                 lambda_lc=1.0, lambda_vc=0.01, lambda_reg=0.1,
                 *args, **kwargs):
        super(RMvLFDAvc, self).__init__(n_components=n_components,
                                        ep_algo=ep_algo,
                                        ep_implementation=ep_implementation,
                                        reg=reg,
                                        kernels=kernels,
                                        affinity_type=affinity_type,
                                        n_neighbors=n_neighbors,
                                        epsilon=epsilon,
                                        affinity_kernel=affinity_kernel,
                                        gamma=gamma,
                                        lambda_lc=lambda_lc, lambda_vc=lambda_vc,
                                        *args, **kwargs)
        self.lambda_reg = lambda_reg
        self.ro = Regularization()

    def calculate_objectives(self):
        self.Sw = self._Sw_() + self.lambda_vc * self.vco.target() + self.lambda_reg * self.ro.target()
        self.Sb = self._Sb_()
