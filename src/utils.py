from itertools import chain
from typing import Iterable, Callable, TypeVar, List

A = TypeVar('A')
T = TypeVar('T')


def flat_map(func: Callable[[A], Iterable[T]], iterable: Iterable[A]) -> List[T]:
    return list(chain.from_iterable(map(func, iterable)))

