from .bases import BaseMvDAlgo
from .objectives import MvDAIntraScatter, MvDAInterScatter, ViewConsistency, Regularization


class MvDA(BaseMvDAlgo):

    def __init__(self,
                 n_components='auto',
                 ep_algo='eigen',
                 ep_implementation='pytorch',
                 reg='auto',
                 kernels=None,
                 *args, **kwargs):
        super(MvDA, self).__init__(n_components=n_components,
                                   ep_algo=ep_algo,
                                   ep_implementation=ep_implementation,
                                   reg=reg,
                                   kernels=kernels,
                                   *args, **kwargs)
        self.swo = MvDAIntraScatter()
        self.sbo = MvDAInterScatter()

    def _Sw_(self):
        return self.swo.target()

    def _Sb_(self):
        return self.sbo.target()


class MvDAvc(MvDA):

    def __init__(self,
                 n_components='auto',
                 ep_algo='eigen',
                 ep_implementation='pytorch',
                 reg='auto',
                 kernels=None,
                 lambda_vc=0.01,
                 *args, **kwargs):
        super(MvDAvc, self).__init__(n_components=n_components,
                                     ep_algo=ep_algo,
                                     ep_implementation=ep_implementation,
                                     reg=reg,
                                     kernels=kernels,
                                     *args, **kwargs)
        self.lambda_vc = lambda_vc
        self.vco = ViewConsistency(reg=reg)

    def calculate_objectives(self):
        self.Sw = self._Sw_() + self.lambda_vc * self.vco.target()
        self.Sb = self._Sb_()


class RMvDA(MvDA):

    def __init__(self,
                 n_components='auto',
                 ep_algo='eigen',
                 ep_implementation='pytorch',
                 reg='auto',
                 kernels=None,
                 lambda_reg=0.1,
                 *args, **kwargs):
        super(RMvDA, self).__init__(n_components=n_components,
                                    ep_algo=ep_algo,
                                    ep_implementation=ep_implementation,
                                    reg=reg,
                                    kernels=kernels,
                                    *args, **kwargs)
        self.lambda_reg = lambda_reg
        self.ro = Regularization()

    def calculate_objectives(self):
        self.Sw = self._Sw_() + self.lambda_reg * self.ro.target()
        self.Sb = self._Sb_()


class RMvDAvc(MvDAvc):

    def __init__(self,
                 n_components='auto',
                 ep_algo='eigen',
                 ep_implementation='pytorch',
                 reg='auto',
                 kernels=None,
                 lambda_vc=0.01, lambda_reg=0.1,
                 *args, **kwargs):
        super(RMvDAvc, self).__init__(n_components=n_components,
                                      ep_algo=ep_algo,
                                      ep_implementation=ep_implementation,
                                      reg=reg,
                                      kernels=kernels,
                                      lambda_vc=lambda_vc,
                                      *args, **kwargs)
        self.lambda_reg = lambda_reg
        self.ro = Regularization()

    def calculate_objectives(self):
        self.Sw = self._Sw_() + self.lambda_vc * self.vco.target() + self.lambda_reg * self.ro.target()
        self.Sb = self._Sb_()
