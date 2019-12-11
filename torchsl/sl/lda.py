from .bases import EOBasedSLAlgo
from .objectives import LDAIntraScatter, LDAInterScatter


class LDA(EOBasedSLAlgo):

    def __init__(self,
                 n_components='auto',
                 ep_algo='eigen',
                 ep_implementation='pytorch',
                 reg='auto',
                 kernel=None,
                 *args, **kwargs):
        super(LDA, self).__init__(n_components=n_components,
                                  ep_algo=ep_algo,
                                  ep_implementation=ep_implementation,
                                  reg=reg,
                                  kernel=kernel,
                                  *args, **kwargs)
        self.swo = LDAIntraScatter()
        self.sbo = LDAInterScatter()

    def _Sw_(self):
        return self.swo.target()

    def _Sb_(self):
        return self.sbo.target()
