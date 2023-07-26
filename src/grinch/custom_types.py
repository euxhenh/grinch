import inspect
from operator import attrgetter
from typing import Any, Dict, List, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt

REP_KEY: TypeAlias = str | List[str] | Dict[str, str] | None
REP: TypeAlias = Dict[str, Any] | List[Any] | Any

NP1D_Any = np.ndarray[Tuple[Any], np.dtype[Any]]
NP2D_Any = np.ndarray[Tuple[Any, Any], np.dtype[Any]]
NP1D_int = np.ndarray[Tuple[Any], np.dtype[np.int_]]
NP2D_int = np.ndarray[Tuple[Any, Any], np.dtype[np.int_]]
NP1D_bool = np.ndarray[Tuple[Any], np.dtype[np.bool_]]
NP2D_bool = np.ndarray[Tuple[Any, Any], np.dtype[np.bool_]]
NP1D_float = np.ndarray[Tuple[Any], np.dtype[np.float_]]
NP2D_float = np.ndarray[Tuple[Any, Any], np.dtype[np.float_]]
NP1D_str = np.ndarray[Tuple[Any], np.dtype[np.str_]]
NP_bool = npt.NDArray[np.bool_]
NP_int = npt.NDArray[np.int_]
NP_float = npt.NDArray[np.float_]


def optional_staticmethod(klas: str, special_args: Dict[str, str]):
    """Marks a method as optionally static. If the method is called from an
    instance of the class, will assume it is not static and will parse any
    self.keys specified in args. Otherwise will set each to None, and call
    the method as a staticmethod. This is useful when static behavior is
    desired if these optional arguments are not always needed.

    This mainly exists because of group_label for group processing where
    optionally we may need to pass group labels. Instead of modifying the
    set_repr function call for evey derived class, we check if the method
    was called from an instance of a class and pass group_label through
    this decorator. This is nice because the derived classes can remain
    ignorant of any group processing details.

    Parameters
    __________
    klas: str
        This should point to the class name of the object.
    args: dict
        Maps a key (member of klas) to a function parameter. These can have
        multiple dots in the representation. E.g., 'foo.bar.eggs.spam' will
        try to access self.foo.bar.eggs.spam
    """
    def _decorator(f):
        def _wrapper(*args, **kwargs):
            bases = inspect.getmro(args[0].__class__)
            base_qualnames = [base.__qualname__ for base in bases]
            if len(args) > 0 and klas in base_qualnames:
                # Assume that this was called from a class instance.
                for k, v in special_args.items():
                    kwargs[v] = attrgetter(k)(args[0])
                args = args[1:]
            return f(*args, **kwargs)
        return _wrapper
    return _decorator
