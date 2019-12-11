from .bases import EOBasedSLAlgo
from .objectives import LFDAIntraScatter, LFDAInterScatter, LFDALLEIntraScatter, LFDALLEInterScatter


class LFDA(EOBasedSLAlgo):

    def __init__(self,
                 n_components='auto',
                 ep_algo='eigen',
                 ep_implementation='pytorch',
                 reg='auto',
                 kernel=None,
                 affinity_type='kernel',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1,
                 *args, **kwargs):
        super(LFDA, self).__init__(n_components=n_components,
                                   ep_algo=ep_algo,
                                   ep_implementation=ep_implementation,
                                   reg=reg,
                                   kernel=kernel,
                                   *args, **kwargs)
        self.swo = LFDAIntraScatter(affinity_type=affinity_type,
                                    n_neighbors=n_neighbors,
                                    epsilon=epsilon,
                                    affinity_kernel=affinity_kernel,
                                    gamma=gamma)
        self.sbo = LFDAInterScatter(affinity_type=affinity_type,
                                    n_neighbors=n_neighbors,
                                    epsilon=epsilon,
                                    affinity_kernel=affinity_kernel,
                                    gamma=gamma)

    def _Sw_(self):
        return self.swo.target()

    def _Sb_(self):
        return self.sbo.target()


class LFDALLE(EOBasedSLAlgo):

    def __init__(self,
                 n_components='auto',
                 ep_algo='eigen',
                 ep_implementation='pytorch',
                 reg='auto',
                 kernel=None,
                 affinity_type='lle',
                 n_neighbors=5,
                 epsilon='auto',
                 affinity_kernel='rbf',
                 gamma=1,
                 lambda_lc=1.0,
                 *args, **kwargs):
        super(LFDALLE, self).__init__(n_components=n_components,
                                      ep_algo=ep_algo,
                                      ep_implementation=ep_implementation,
                                      reg=reg,
                                      kernel=kernel,
                                      *args, **kwargs)
        self.swo = LFDALLEIntraScatter(affinity_type=affinity_type,
                                       n_neighbors=n_neighbors,
                                       epsilon=epsilon,
                                       affinity_kernel=affinity_kernel,
                                       gamma=gamma,
                                       lambda_lc=lambda_lc)
        self.sbo = LFDALLEInterScatter(affinity_type=affinity_type,
                                       n_neighbors=n_neighbors,
                                       epsilon=epsilon,
                                       affinity_kernel=affinity_kernel,
                                       gamma=gamma,
                                       lambda_lc=lambda_lc)

    def _Sw_(self):
        return self.swo.target()

    def _Sb_(self):
        return self.sbo.target()
