from .bases import BaseMvDAlgo
from .objectives import MvDAInterScatter, MvDAIntraScatter, ClassSeparating


class MvCSDA(BaseMvDAlgo):

    def __init__(self,
                 n_components='auto',
                 ep_algo='eigen',
                 ep_implementation='pytorch',
                 reg='auto',
                 kernels=None,
                 *args, **kwargs):
        super(MvCSDA, self).__init__(n_components=n_components,
                                     ep_algo=ep_algo,
                                     ep_implementation=ep_implementation,
                                     reg=reg,
                                     kernels=kernels,
                                     *args, **kwargs)
        self.swo = MvDAIntraScatter()
        self.cso = ClassSeparating()

    def _Sw_(self):
        return self.swo.target()

    def _Sb_(self):
        return self.cso.target()


class MvDAplusCS(BaseMvDAlgo):

    def __init__(self,
                 n_components='auto',
                 ep_algo='eigen',
                 ep_implementation='pytorch',
                 reg='auto',
                 kernels=None,
                 lambda_cs=1.,
                 *args, **kwargs):
        super(MvDAplusCS, self).__init__(n_components=n_components,
                                         ep_algo=ep_algo,
                                         ep_implementation=ep_implementation,
                                         reg=reg,
                                         kernels=kernels,
                                         *args, **kwargs)
        self.lambda_cs = lambda_cs
        self.swo = MvDAIntraScatter()
        self.sbo = MvDAInterScatter()
        self.cso = ClassSeparating()

    def _Sw_(self):
        return self.swo.target()

    def _Sb_(self):
        return self.sbo.target() + self.lambda_cs * self.cso.target()
