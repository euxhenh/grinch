# mypy: disable-error-code = used-before-def

import abc
import inspect
import logging
from functools import wraps
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Annotated,
    Callable,
    Dict,
    List,
    TypeAlias,
    TypeVar,
)

from anndata import AnnData
from pydantic import Field, field_validator, validate_call

from ..base import StorageMixin
from ..conf import BaseConfigurable
from ..custom_types import NP1D_int
from ..utils.ops import safe_format
from ..utils.validation import all_not_None

logger = logging.getLogger(__name__)


T = TypeVar('T')
# Parameter that will be passed to the underlying processor.
ProcessorParam = Annotated[T, 'ProcessorParam']

# Storage and retrieval keys
ReadKey: TypeAlias = str
WriteKey: TypeAlias = str


class BaseProcessor(BaseConfigurable, StorageMixin):
    r"""A base class for all processors. A processor cannot update the
    data matrix X, but can use it to perform any kind of fitting. The
    processor is in charge of resolving all reads and writes in the AnnData
    object. It does so by taking as input string key(s) that point to the
    adata column/key that will be used for reading or writing. These keys
    are defined inside the Config of the derived class and are evaluated at
    initialization to conform with accepted adata columns.

    This class also implements a 'processor' property which should point to
    a wrapped processor object (if any). E.g., the processor can point to
    sklearn's implementations of estimators.

    Class Attributes
    ----------------
    __processor_attrs__ : List[str]
        A list of processor attributes to save along with results after
        fitting.

    __processor_reqs__ : List[str]
        A list of methods that the processor should implement. Raises an
        error if any not found.

    Attributes
    ----------
    processor : Any
        The underlying processor used for data fitting, transformation etc.
    """
    __slots__ = ['_processor']

    __buffers__ = ['_processor']
    __processor_attrs__: List[str] = []
    __processor_reqs__: List[str] = []

    class Config(BaseConfigurable.Config):
        r"""BaseProcessor.Config

        Parameters
        ----------
        attrs_key : str, default=None
            The key to store processors attributes in (post fit). Curly
            brackets will be formatted. E.g., if a Predictor has a key
            called `labels_key`, one can set `attrs_key` to
            `uns.{labels_key}_`. The value (tail) of `labels_key` will
            automatically be parsed into `attrs_key`.

        kwargs : dict, default={}
            Any additional processor parameters should be specified under a
            `kwargs` dict. Important parameters may be explicitly set in
            the Config for convenience.

        Class attributes
        ----------------
        __extra_processor_params__ : List[str]
            Kwargs used by the processor, but are not ProcessorParam's or
            have a different name than the one specified in the Config.
            E.g., `random_state` should be here, as we use `seed`. The only
            use for this list is to remove such keys from `kwargs` if any
            is present.
        """
        if TYPE_CHECKING:
            create: Callable[..., 'BaseProcessor']

        attrs_key: WriteKey | None = None
        kwargs: Dict[str, ProcessorParam] = Field(default_factory=dict)

        __extra_processor_params__: List[str] = []

        def model_post_init(self, __context):
            """Safely formats attrs key using any field that is a str."""
            super().model_post_init(__context)
            if self.attrs_key is None:
                return
            # Use only the tail of other keys to format attrs_key.
            field_dict = {
                k: v.rsplit('.', 1)[-1] for k, v in self.model_dump().items()
                if isinstance(v, str)
            }
            self.attrs_key = safe_format(self.attrs_key, **field_dict)

        @field_validator('kwargs')
        def remove_explicit_args(cls, val):
            """Remove ProcessorParam's if present in kwargs to avoid
            duplication.
            """
            for explicit_key in chain(cls.processor_params(),
                                      cls.__extra_processor_params__):
                if val.pop(explicit_key, None) is not None:
                    logger.warning(
                        f"Popping '{explicit_key=}' from kwargs. "
                        "This key has been set in the Config."
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

    @property
    def processor(self):
        """Returns the object that is being wrapped by the class."""
        return getattr(self, '_processor', None)

    @processor.setter
    def processor(self, value):
        """Sets the processor and checks if it implements any methods
        required by each parent class.
        """
        for cls in inspect.getmro(self.__class__):
            if not hasattr(cls, '__processor_reqs__'):
                continue
            for method_name in cls.__processor_reqs__:
                method = getattr(value, method_name, None)
                if not callable(method):
                    raise ValueError(
                        f"Object of type '{type(value)}' does not implement "
                        f"a callable method named '{method_name}'."
                    )
        self._processor = value

    @classmethod
    @wraps(Config.processor_params)
    def processor_params(cls) -> List[str]:
        """Same as self.cfg.processor_params"""
        return cls.Config.processor_params()

    @StorageMixin.lazy_writer
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(
        self,
        adata: AnnData,
        *,
        obs_indices: NP1D_int | None = None,
        var_indices: NP1D_int | None = None,
        **kwargs,
    ):
        """Calls the processor with adata.

        Order of operations is as follows:
        1) Clear self.storage
        2) Run this __call__
            a) Index into obs or var if any.
            b) Run preprocessing
            c) Run processing
            d) Run postprocessing
            e) Store processor attributes
        3) Return storage if kwargs['return_storage'] or write otherwise
        """
        if all_not_None(obs_indices, var_indices):
            adata = adata[obs_indices, var_indices]
        elif obs_indices is not None:
            # Storage is applied to the full adata from the decorator
            adata = adata[obs_indices]
        elif var_indices is not None:
            adata = adata[:, var_indices]

        self._pre_process(adata)
        self._process(adata)
        self._post_process(adata)
        self.store_attrs()

    def _pre_process(self, adata: AnnData) -> None:
        """Will run before `self._process`.

        Useful for classes that perform additional pre-processing, but do
        not wish to modify the `_process` step of the parent.
        """
        return

    @abc.abstractmethod
    def _process(self, adata: AnnData) -> None:
        """To be implemented by a derived class."""
        raise NotImplementedError

    def _post_process(self, adata: AnnData) -> None:
        """Will run after `self._process`.

        Useful for classes that perform additional post-processing, but do
        not wish to modify the `_process` step of the parent.
        """
        return

    def store_attrs(self) -> None:
        """Save processor attributes under `cfg.attrs_key` if any."""
        if None in (self.processor, self.cfg.attrs_key) or not self.__processor_attrs__:
            return
        attrs = {at: getattr(self.processor, at) for at in self.__processor_attrs__}
        assert self.cfg.attrs_key is not None  # for mypy
        self.store_item(self.cfg.attrs_key, attrs)
