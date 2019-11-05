from .bases import MvDAlgoBase
from .objectives import MvDAIntraScatter, MvDAInterScatter, ViewConsistency, Regularization


class MvDA(MvDAlgoBase):
    name = 'Multi-view Discriminant Analysis'

    def __init__(self,
                 n_components='auto',
                 ep='eig',
                 reg='auto',
                 kernels=None):
        super(MvDA, self).__init__(n_components=n_components,
                                   ep=ep,
                                   reg=reg,
                                   kernels=kernels)
        self.swo = MvDAIntraScatter()
        self.sbo = MvDAInterScatter()

    def _Sw_(self):
        return self.swo.target()

    def _Sb_(self):
        return self.sbo.target()


class MvDAvc(MvDA):
    name = MvDA.name + ' with View-Consistency'

    def __init__(self,
                 n_components='auto',
                 ep='eig',
                 reg='auto',
                 kernels=None,
                 lambda_vc=0.01):
        super(MvDAvc, self).__init__(n_components=n_components,
                                     ep=ep,
                                     reg=reg,
                                     kernels=kernels)
        self.lambda_vc = lambda_vc
        self.vco = ViewConsistency(reg=reg)

    def calculate_objectives(self):
        self.Sw = self._Sw_() + self.lambda_vc * self.vco.target()
        self.Sb = self._Sb_()


class RMvDA(MvDA):
    name = 'Regularized ' + MvDA.name

    def __init__(self,
                 n_components='auto',
                 ep='eig',
                 reg='auto',
                 kernels=None,
                 lambda_reg=0.1):
        super(RMvDA, self).__init__(n_components=n_components,
                                    ep=ep,
                                    reg=reg,
                                    kernels=kernels)
        self.lambda_reg = lambda_reg
        self.ro = Regularization()

    def calculate_objectives(self):
        self.Sw = self._Sw_() + self.lambda_reg * self.ro.target()
        self.Sb = self._Sb_()


class RMvDAvc(MvDAvc):
    name = 'Regularized ' + MvDAvc.name

    def __init__(self,
                 n_components='auto',
                 ep='eig',
                 reg='auto',
                 kernels=None,
                 lambda_vc=0.01, lambda_reg=0.1):
        super(RMvDAvc, self).__init__(n_components=n_components,
                                      ep=ep,
                                      reg=reg,
                                      kernels=kernels,
                                      lambda_vc=lambda_vc)
        self.lambda_reg = lambda_reg
        self.ro = Regularization()

    def calculate_objectives(self):
        self.Sw = self._Sw_() + self.lambda_vc * self.vco.target() + self.lambda_reg * self.ro.target()
        self.Sb = self._Sb_()
