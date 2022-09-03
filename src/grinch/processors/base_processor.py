import abc
import inspect
import logging
from functools import partial
from itertools import islice, starmap
from operator import itemgetter
from typing import Any, Callable, Dict, List, Optional, Tuple

from anndata import AnnData
from pydantic import validate_arguments, validator

from ..aliases import ALLOWED_KEYS
from ..conf import BaseConfigurable
from ..custom_types import REP, REP_KEY, optional_staticmethod
from ..utils.adata import as_empty
from ..utils.ops import compose, safe_format
from ..utils.validation import check_has_processor

logger = logging.getLogger(__name__)


def adata_modifier(f: Callable):
    """A decorator for lazy adata setattr. This exists so that all
    BaseProcessors don't set any adata keys themselves and everything is
    handled by the BaseProcessor class. This is useful for GroupProcessing,
    saving adata without the data matrix, etc.
    """
    params = inspect.signature(f).parameters
    # require 'self' and 'adata'
    if len(params) < 2:
        raise ValueError("A 'setter_method' should take at least 2 arguments.")

    _, adata_parameter = next(islice(params.items(), 1, 2))
    if not issubclass(AnnData, adata_parameter.annotation):
        raise ValueError(
            "First argument to a 'setter_method' should be of explicit type AnnData."
        )

    def _wrapper(
        self,
        adata: AnnData,
        *args, **kwargs
    ) -> Optional[AnnData | Dict[str, REP] | Tuple[AnnData, Dict[str, REP]]]:
        # init empty storage dict
        self.storage = {}
        outp = f(self, adata, *args, **kwargs)
        if outp is not None:
            logger.warning(
                f"Function '{f.__name__}' returned a value. This will be discarded."
            )
        # no_data_matrix can be useful e.g., in group mode when adatas
        # are split into groups and we don't want to unnecessarily
        # create actual copies as they will be concatenated anyway.
        # 'no_data_matrix' should be keyword only
        if 'no_data_matrix' in kwargs:
            adata = as_empty(adata) if kwargs['no_data_matrix'] else adata

        return_storage = 'return_storage' in kwargs and kwargs['return_storage']

        if hasattr(self.cfg, 'save_stats') and self.cfg.save_stats:
            self.save_processor_stats()
        if len(self.storage) > 0 and not return_storage:
            self.set_repr(adata, list(self.storage), list(self.storage.values()))

        if not self.cfg.inplace and return_storage:
            return adata, self.storage
        elif return_storage:
            return self.storage
        elif not self.cfg.inplace:
            return adata
        return None

    return _wrapper


class BaseProcessor(BaseConfigurable):
    """A base class for dimensionality reduction, clustering and other
    processors. A processor cannot update the data matrix X, but can use it
    to perform any kind of fitting. The processor is in charge of resolving
    all reads and writes in the AnnData object. It does so by taking as
    input string key(s) that point to the adata column/key that will be
    used for reading or writing. These keys are defined inside the Config
    of the derived class and are evaluated at initialization to conform
    with accepted adata columns.

    This class also implements a 'processor' property which should point to
    a wrapped processor object (if any). E.g., the processor can point to
    sklearn's implementations of estimators.

    'get_repr' and 'set_repr' are like smarter versions of getattr and
    setattr in Python which also handle adata columns. They use getattr and
    setattr internally. These repr functions support keys that are single
    strings, list of strings, or dictionaries of strings to strings. In the
    latter case, the keys are used to index the value being set (must be a
    dict), and the values should point to the adata column/key to use for
    storing.

    Parameters
    __________
    inplace: bool
        If False, will make and return a copy of adata.
    *_key: custom_types.REP_KEY
        Any Config member parameter that ends in '_key' will be checked by
        pydantic validators to conform with adata column names.
    read_key_prefix, save_key_prefix: str
        Will prepend this prefix to all (last) read/save_keys. This is useful,
        for example, for GroupProcess, which prepends the 'group{label}'
        prefix to all read/saved reps.
    """

    class Config(BaseConfigurable.Config):
        inplace: bool = True
        read_key_prefix: str = ''
        save_key_prefix: str = ''

        @staticmethod
        def _validate_single_rep_key(val: str):
            """Validates the format of a single key (str)."""
            if val is None or val in ['X', 'obs_names', 'var_names']:
                return val
            if '.' not in val:
                raise ValueError(
                    "Representation keys must equal 'X' or must contain a "
                    "dot '.' that points to the AnnData column to use."
                )
            if len(parts := val.split('.')) > 2 and parts[0] != 'uns':
                raise ValueError("There can only be one dot '.' in non-uns representation keys.")
            if parts[0] not in ALLOWED_KEYS:
                raise ValueError(f"AnnData annotation key should be one of {ALLOWED_KEYS}.")
            if len(parts[1]) >= 120:
                raise ValueError("Columns keys should be less than 120 characters.")
            return val

        @validator('*')
        def rep_format_is_correct(cls, val, field):
            """Select the representation to use. If val is str: if 'X',
            will use adata.X, otherwise it must contain a dot that splits
            the annotation key that will be used and the column key. E.g.,
            'obsm.x_emb' will use 'adata.obsm['x_emb']'. A list of str will
            be parsed as *args, and a dict of (str, str) should map a
            dictionary key to the desired representation. The latter is
            useful when calling, for example, predictors which require a
            data representation X and labels y. In this case, X and y would
            be dictionary keys and the corresponding representations for X
            and y would be the values.

            This validator will only check fields that end with '_key'.
            """
            if not field.name.endswith('_key'):
                return val

            match val:
                case str() as v:
                    return cls._validate_single_rep_key(v)
                case [*vals]:
                    return [cls._validate_single_rep_key(v) for v in vals]
                case {**vals}:
                    return {k: cls._validate_single_rep_key(v) for k, v in vals.items()}
                case None:
                    return None
                case _:
                    raise ValueError(
                        f"Could not interpret format for {field.name}. Please make sure "
                        "it is a str, list[str], or dict[str, str]."
                    )

        def get_save_key_prefix(
            self,
            current_prefix: str,
            splitter: str = '.',
            **kwargs
        ) -> str:
            """Returns a save_key_prefix. Can be used to search recursively
            for save_key prefixes.
            """
            upstream_prefix = self.save_key_prefix
            if len(upstream_prefix) > 0:  # happens when GroupProcess modules are stacked
                upstream_prefix += splitter
            current_prefix = safe_format(current_prefix, **kwargs)
            prefix = f'{upstream_prefix}{current_prefix}'
            return prefix

    cfg: Config

    def __init__(self, cfg: Config, /):
        super().__init__(cfg)

        self.storage: Dict[str, REP] = {}

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

    def save_processor_stats(self) -> None:
        check_has_processor(self)
        if not hasattr(self.cfg, 'stats_key'):
            raise KeyError(f"No 'stats_key' was found in {self.cfg.__class__.__qualname__}.")
        # Assume it has been explicitly set to None
        if self.cfg.stats_key is None:  # type: ignore
            return

        stats = {stat: getattr(self.processor, stat) for stat in self._processor_stats()}
        if stats:
            self.store_item(self.cfg.stats_key, stats)  # type: ignore

    @adata_modifier
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __call__(
        self,
        adata: AnnData,
        *,
        no_data_matrix: bool = False,
        return_storage: bool = False
    ):
        """Calls the processor with adata. Will copy adata if inplace was
        set to False.
        """
        if not self.cfg.inplace and not no_data_matrix:
            adata = adata.copy()

        self._process(adata)

    @abc.abstractmethod
    def _process(self, adata: AnnData) -> None:
        """To be implemented by a derived class."""
        raise NotImplementedError

    def store_item(self, key: str, val: REP) -> None:
        """Will store the value to a key for lazy saving into adata."""
        self.storage[key] = val

    @staticmethod
    def __prepend_to_last_key(key: str, prefix: str):
        pref_keys, last_key = key.rsplit('.', maxsplit=1)
        return f'{pref_keys}.{prefix}{last_key}'

    @staticmethod
    def _get_repr(adata: AnnData, key: str, read_key_prefix: str = '') -> Any:
        """Get the data representation that key points to."""
        if key is None:
            raise ValueError("Cannot get representation if 'key' is None.")

        if key == 'X':
            return adata.X
        if key in ['obs_names', 'var_names']:
            return getattr(adata, key).to_numpy().astype(str)

        key = BaseProcessor.__prepend_to_last_key(key, read_key_prefix)

        read_class, *read_keys = key.split('.')
        # We only support dictionary style access for read_keys
        rec_itemgetter = compose(*(itemgetter(rk) for rk in read_keys))
        klas = getattr(adata, read_class)
        return rec_itemgetter(klas)

    @optional_staticmethod('BaseProcessor', {'cfg.read_key_prefix': 'read_key_prefix'})
    def get_repr(adata: AnnData, key: REP_KEY, read_key_prefix: str = '') -> REP:
        """Get the representation(s) that read_key points to."""
        single_get_func = partial(
            BaseProcessor._get_repr,
            adata,
            read_key_prefix=read_key_prefix,
        )
        match key:
            case str() as v:
                return single_get_func(v)
            case [*vals]:
                return [single_get_func(v) for v in vals]
            case {**vals}:
                return {k: single_get_func(v) for k, v in vals.items()}  # type:ignore
            case _:
                raise ValueError(f"'{key}' format not understood.")

    @staticmethod
    def _set_repr(adata: AnnData, key: str, value: Any, save_key_prefix: str = ''):
        """Save value under the key pointed to by key.
        """
        if key is None:
            raise ValueError("Cannot save representation if 'key' is None.")
        # Add prefix to the last save key
        key = BaseProcessor.__prepend_to_last_key(key, save_key_prefix)

        save_class, *save_keys = key.split('.')
        klas = getattr(adata, save_class)
        # Iterate over all save keys and initialize empty dictionaries if
        # the keys are not found.
        while len(save_keys) > 1:
            save_key = save_keys.pop(0)
            if save_key not in klas:
                klas[save_key] = {}
            klas = klas[save_key]
        # Final key
        save_key = save_keys.pop(0)
        assert len(save_keys) == 0
        klas[save_key] = value

    @optional_staticmethod('BaseProcessor', {'cfg.save_key_prefix': 'save_key_prefix'})
    def set_repr(
        adata: AnnData,
        key: REP_KEY,
        value: REP,
        save_key_prefix: str = ''
    ) -> None:
        single_set_func = partial(
            BaseProcessor._set_repr,
            adata,
            save_key_prefix=save_key_prefix,
        )
        """Saves values under the key that save_key points to. Not to be
        called by any derived classes."""
        match key, value:
            # Match a string key and Any value
            case str() as key, val:
                single_set_func(key, val)
            # Match a list of keys and a list of vals
            case [*keys], [*vals]:
                if len(keys) != len(vals):
                    raise ValueError("Inconsistent length between save_key and value.")
                _ = list(starmap(single_set_func, zip(keys, vals)))
            # Match a dict of keys and a dict of vals
            case {**keys}, {**vals}:
                # Make sure all keys exist
                keys_not_found = set(keys).difference(vals)
                if len(keys_not_found) > 0:
                    raise ValueError(
                        f"Keys {keys_not_found} were not found in the output dictionary."
                    )
                _ = list(starmap(single_set_func, ((v, vals[k]) for k, v in keys.items())))
            # No match
            case _:
                raise ValueError(
                    f"Inconsistent format between value and key. Key has type {type(key)} "
                    f"but value has type {type(value)}."
                )
