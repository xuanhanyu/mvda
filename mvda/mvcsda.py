from .bases import MvDAlgoBase
from .objectives import MvDAInterScatter, MvDAIntraScatter, ClassSeparating


class MvCSDA(MvDAlgoBase):
    name = 'Multi-view Discriminant Analysis with Class Separation'

    def __init__(self,
                 n_components='auto',
                 ep='eig',
                 reg='auto',
                 kernels=None,
                 *args, **kwargs):
        super(MvCSDA, self).__init__(n_components=n_components,
                                     ep=ep,
                                     reg=reg,
                                     kernels=kernels,
                                     *args, **kwargs)
        self.swo = MvDAIntraScatter()
        self.cso = ClassSeparating()

    def _Sw_(self):
        return self.swo.target()

    def _Sb_(self):
        return self.cso.target()


class MvDAplusCS(MvDAlgoBase):
    name = 'Multi-view Discriminant Analysis plus Class Separation'

    def __init__(self,
                 n_components='auto',
                 ep='eig',
                 reg='auto',
                 kernels=None,
                 lambda_cs=1.,
                 *args, **kwargs):
        super(MvDAplusCS, self).__init__(n_components=n_components,
                                         ep=ep,
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
