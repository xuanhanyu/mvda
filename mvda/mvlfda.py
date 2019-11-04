from .mvdbase import MvDAlgoBase
from .objectives import MvLFDAIntraScatter, MvLFDAInterScatter


class MvLFDA(MvDAlgoBase):
    name = 'Multi-view Discriminant Analysis'

    def __init__(self, n_components='auto', ep='eig', reg='auto'):
        super(MvLFDA, self).__init__(n_components, ep, reg)
        self.swo = MvLFDAIntraScatter()
        self.sbo = MvLFDAInterScatter()
        self.add_dependents_(self.swo, self.sbo)
        self.add_dependencies_(self.swo, self.sbo)

    def __Sw__(self):
        return self.swo.target()

    def __Sb__(self):
        return self.sbo.target()
