# mypy: disable-error-code = used-before-def

import abc
import inspect
import logging
from itertools import chain, islice
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    TypeAlias,
    TypeVar,
)

from anndata import AnnData
from pydantic import field_validator, validate_call

from ..base import StorageMixin
from ..conf import BaseConfigurable
from ..custom_types import NP1D_int
from ..utils.validation import all_not_None, check_has_processor

logger = logging.getLogger(__name__)


T = TypeVar('T')
# Parameter that will be passed to the underlying processor.
ProcessorParam = Annotated[T, 'ProcessorParam']

# Storage and retrieval keys
ReadKey: TypeAlias = str
WriteKey: TypeAlias = str


def adata_modifier(f: Callable):
    """A decorator for lazy adata setattr. This exists so that all
    BaseProcessors don't set any adata keys themselves and everything is
    handled by the BaseProcessor class. This is useful for GroupProcessing.
    """
    params = inspect.signature(f).parameters
    # require 'self' and 'adata'
    if len(params) < 2:
        raise ValueError("A 'setter_method' should take at least 2 arguments.")

    _, adata_parameter = next(islice(params.items(), 1, 2))
    if not issubclass(AnnData, adata_parameter.annotation):
        raise ValueError(
            "First argument to a 'setter_method' should be explicitly typed 'AnnData'."
        )

    def _wrapper(self: 'BaseProcessor', adata: AnnData, *args, **kwargs) -> Dict[str, Any] | None:
        # init empty storage dict
        self._storage: Dict[str, Any] = {}
        outp = f(self, adata, *args, **kwargs)
        if outp is not None:
            logger.warning(
                f"Function '{f.__name__}' returned a value. This will be discarded."
            )

        self.save_stats()

        return_storage: bool = kwargs.get('return_storage', False)
        if not return_storage and len(self._storage) > 0:
            self.set_repr(
                adata,
                list(self._storage),
                list(self._storage.values())
            )
        else:
            return self._storage
        return None

    return _wrapper


class BaseProcessor(BaseConfigurable, StorageMixin):
    """A base class for all processors. A processor cannot update the
    data matrix X, but can use it to perform any kind of fitting. The
    processor is in charge of resolving all reads and writes in the AnnData
    object. It does so by taking as input string key(s) that point to the
    adata column/key that will be used for reading or writing. These keys
    are defined inside the Config of the derived class and are evaluated at
    initialization to conform with accepted adata columns.

    This class also implements a 'processor' property which should point to
    a wrapped processor object (if any). E.g., the processor can point to
    sklearn's implementations of estimators.

    Attributes
    ----------
    __stats__: List[str]
        A list of processor attributes to save along with results after
        fitting.

    _storage : Dict[str, Any]
        A dict mapping a key to a representation. Is used internally.
    """
    __stats__: List[str] = []  # processor attributes to store along results

    class Config(BaseConfigurable.Config):
        r"""BaseProcessor.Config

        Parameters
        ----------
        inplace : bool, default=True
            If False, will make and return a copy of adata.

        kwargs : dict, default={}
            Any Processor parameters that should be passed to the inner
            processor object (or related methods).

        Class attributes
        ----------------
        __other_processor_params__ : List[str]
            Holds kwargs that will be passed to the underlying
            processor/processor function, but are not marked as
            ProcessorParams and are also not passed via kwargs.
        """
        if TYPE_CHECKING:
            create: Callable[..., 'BaseProcessor']

        inplace: bool = True
        # Processor kwargs
        kwargs: Dict[str, ProcessorParam] = {}

        # Kwargs used by the processor, but are not ProcessorParam's
        __other_processor_params__: List[str] = []

        @field_validator('kwargs')
        def remove_explicit_args(cls, val):
            """Remove ProcessorParam's if present in kwargs to avoid
            duplication.
            """
            processor_params = cls.processor_params()
            for explicit_key in chain(processor_params, cls.__other_processor_params__):
                if val.pop(explicit_key, None) is not None:
                    logger.warning(
                        f"Popping '{explicit_key}' from kwargs. "
                        "This key has been set explicitly."
                    )
            return val

        @classmethod
        def processor_params(cls) -> List[str]:
            """Get all ProcessorParam's defined in this Config.

            Returns
            -------
            params : List[str]
                A list of parameter names.
            """
            params: List[str] = []
            for param, field_info in cls.model_fields.items():
                if 'ProcessorParam' in field_info.metadata:
                    params.append(param)
            return params

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self._storage: Dict[str, Any] = {}

    @property
    def processor(self):
        """Points to the object that is being wrapped by the derived class.
        Present for consistency among derived classes. Returns None if a
        processor has not been assigned.
        """
        return getattr(self, '_processor', None)

    @processor.setter
    def processor(self, value):
        """Sets the processor and checks if it implements any required methods.
        """
        for method_name in self._processor_must_implement():
            method = getattr(value, method_name, None)
            if not callable(method):
                raise ValueError(
                    f"Object of type '{type(value)}' does not implement "
                    f"a callable '{method_name}' method."
                )
        self._processor = value

    @classmethod
    def processor_params(cls) -> List[str]:
        return cls.Config.processor_params()

    @staticmethod
    def _processor_must_implement() -> List[str]:
        """Upon assignment, will check if the processor contains the fields
        contained in this list.
        """
        return []

    @staticmethod
    def _processor_stats() -> List[str]:
        """Processor attributes to extract into a stats dictionary for writing.
        """
        return []

    def save_stats(self) -> None:
        if len(self.__stats__) == 0:
            return

        check_has_processor(self)
        if not hasattr(self.cfg, 'stats_key'):
            raise KeyError(
                "No 'stats_key' was found "
                f"in {self.cfg.__class__.__qualname__}."
            )
        # Assume it has been explicitly set to None
        if self.cfg.stats_key is None:  # type: ignore
            return

        stats = {stat: getattr(self.processor, stat)
                 for stat in self._processor_stats()}
        if stats:
            self.store_item(self.cfg.stats_key, stats)  # type: ignore

    @adata_modifier
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(
        self,
        adata: AnnData,
        *,
        obs_indices: NP1D_int | None = None,
        var_indices: NP1D_int | None = None,
        **kwargs,
    ):
        """Calls the processor with adata. Will copy adata if inplace was
        set to False.
        """
        if all_not_None(obs_indices, var_indices):
            adata = adata[obs_indices, var_indices]
        elif obs_indices is not None:
            # Storage is applied to the full adata from the decorator
            adata = adata[obs_indices]
        elif var_indices is not None:
            adata = adata[:, var_indices]

        self._process(adata)

    @abc.abstractmethod
    def _process(self, adata: AnnData) -> None:
        """To be implemented by a derived class."""
        raise NotImplementedError
