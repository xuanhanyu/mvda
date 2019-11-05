try:
    matlab_engine = __import__('matlab.engine')
    del matlab_engine
except ModuleNotFoundError:
    print('[Warning] Matlab interface not found.'
          'Install it to use LDAX Eigen Solver.')

from .mvda import MvDA, MvDAvc, RMvDA, RMvDAvc
from .mvcsda import MvCSDA, MvDAplusCS
from .mvlfda import MvLFDA, MvLFDAvc, RMvLFDA, RMvLFDAvc
from .mvccda import MvCCDA, MvDCCCDA


__all__ = [
    'MvDA', 'MvDAvc', 'RMvDA', 'RMvDAvc',
    'MvCSDA', 'MvDAplusCS',
    'MvLFDA', 'MvLFDAvc', 'RMvLFDA', 'RMvLFDAvc',
    'MvCCDA', 'MvDCCCDA'
]
