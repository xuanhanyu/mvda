from .mvdbase import MvDAlgoBase
from .objectives import MvDAIntraScatter, MvDAInterScatter, ViewConsistency


class MvDA(MvDAlgoBase):
    name = 'Multi-view Discriminant Analysis'

    def __init__(self, n_components='auto', ep='eig', reg='auto'):
        super(MvDA, self).__init__(n_components, ep, reg)
        self.swo = MvDAIntraScatter()
        self.sbo = MvDAInterScatter()
        self.add_dependents_(self.swo, self.sbo)
        self.add_dependencies_(self.swo, self.sbo)

    def __Sw__(self):
        return self.swo.target()

    def __Sb__(self):
        return self.sbo.target()


class MvDAvc(MvDA):
    name = 'Multi-view Discriminant Analysis with View-Consistency'

    def __init__(self, n_components='auto', ep='eig', reg='auto', lambda_vc=0.01):
        super(MvDAvc, self).__init__(n_components, ep, reg)
        self.lambda_vc = lambda_vc
        self.vco = ViewConsistency(reg=reg)
        self.add_dependents_(self.vco)
        self.add_dependencies_(self.vco)

    def calculate_objectives(self):
        self.Sw = self.__Sw__() + self.lambda_vc * self.vco.target()
        self.Sb = self.__Sb__()
