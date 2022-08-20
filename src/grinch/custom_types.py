from typing import Any, Dict, List, Optional, TypeAlias, Tuple

import numpy as np
import numpy.typing as npt

REP_KEY: TypeAlias = Optional[str | List[str] | Dict[str, str]]
REP: TypeAlias = Dict[str, Any] | List[Any] | Any

NP1D_Any = np.ndarray[Tuple[Any], np.dtype[Any]]
NP2D_Any = np.ndarray[Tuple[Any, Any], np.dtype[Any]]
NP1D_int = np.ndarray[Tuple[Any], np.int_]
NP2D_int = np.ndarray[Tuple[Any, Any], np.int_]
NP_bool = npt.NDArray[np.bool_]
NP_int = npt.NDArray[np.int_]
