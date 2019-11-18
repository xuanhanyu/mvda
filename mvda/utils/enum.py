from enum import Enum, EnumMeta
from typing import Any, Iterable


class PredefinedArgumentsMeta(EnumMeta):
    def __contains__(cls: Iterable[Any], item: Any):
        for _ in cls:
            if _ == item:
                return True

    def __getitem__(cls, item):
        for _ in cls:
            if item == _:
                return _

    # def __iter__(cls):
    #     return (cls._.name for _ in cls)


class PredefinedArguments(Enum, metaclass=PredefinedArgumentsMeta):

    def __eq__(self, other):
        return super() == other or self.value == other or other in self.value

