from typing import Annotated, Any, Dict, List, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from pydantic import Field

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

NP_SP = np.ndarray | sp.spmatrix

PercentFraction = Annotated[float, Field(ge=0, le=1)]
