from .mvdbase import MvDAlgoBase
from .objectives import MvDAInterScatter, MvDAIntraScatter, ClassSeparating


class MvDAcs(MvDAlgoBase):
    name = 'Multi-view Discriminant Analysis with Class Separation'

    def __init__(self, n_components='auto', ep='eig', reg='auto'):
        super(MvDAcs, self).__init__(n_components, ep, reg)
        self.swo = MvDAIntraScatter()
        self.cso = ClassSeparating()
        self.add_dependents_(self.swo, self.cso)
        self.add_dependencies_(self.swo, self.cso)

    def __Sw__(self):
        return self.swo.target()

    def __Sb__(self):
        return self.cso.target()


class MvDAplusCS(MvDAlgoBase):
    name = 'Multi-view Discriminant Analysis plus Class Separation'

    def __init__(self, n_components='auto', ep='eig', reg='auto', lambda_cs=1.):
        super(MvDAplusCS, self).__init__(n_components, ep, reg)
        self.lambda_cs = lambda_cs
        self.swo = MvDAIntraScatter()
        self.sbo = MvDAInterScatter()
        self.cso = ClassSeparating()
        self.add_dependents_(self.swo, self.sbo, self.cso)
        self.add_dependencies_(self.swo, self.sbo, self.cso)

    def __Sw__(self):
        return self.swo.target()

    def __Sb__(self):
        return self.sbo.target() + self.lambda_cs * self.cso.target()
