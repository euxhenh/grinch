# mypy: disable-error-code = used-before-def

import abc
import inspect
import logging
from functools import partial
from itertools import islice, starmap
from operator import itemgetter
from typing import Any, Callable, Dict, List

from anndata import AnnData
from pydantic import field_validator, validate_call

from ..aliases import ALLOWED_KEYS
from ..conf import BaseConfigurable
from ..custom_types import REP, REP_KEY, NP1D_int
from ..utils.ops import compose, safe_format
from ..utils.validation import all_not_None, check_has_processor

logger = logging.getLogger(__name__)


def adata_modifier(f: Callable):
    """A decorator for lazy adata setattr. This exists so that all
    BaseProcessors don't set any adata keys themselves and everything is
    handled by the BaseProcessor class. This is useful for GroupProcessing.
    """
    params = inspect.signature(f).parameters
    # require 'self' and 'adata'
    if len(params) < 2:
        raise ValueError(
            "A 'setter_method' should take at least 2 arguments."
        )

    _, adata_parameter = next(islice(params.items(), 1, 2))
    if not issubclass(AnnData, adata_parameter.annotation):
        raise ValueError(
            "First argument to a 'setter_method' "
            "should be of explicit type AnnData."
        )

    def _wrapper(
        self: 'BaseProcessor', adata: AnnData, *args, **kwargs
    ) -> Dict[str, REP] | None:
        # init empty storage dict
        self.storage = {}
        outp = f(self, adata, *args, **kwargs)
        if outp is not None:
            logger.warning(
                f"Function '{f.__name__}' returned a value."
                " This will be discarded."
            )
        if hasattr(self.cfg, 'save_stats') and self.cfg.save_stats:
            self.save_processor_stats()

        return_storage: bool = kwargs.get('return_storage', False)
        if not return_storage and len(self.storage) > 0:
            self.set_repr(
                adata,
                list(self.storage),
                list(self.storage.values())
            )
        else:
            return self.storage
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
        Will prepend this prefix to all (last) read/save_keys. This is
        useful, for example, for GroupProcess, which prepends the
        'group{label}' prefix to all read/saved reps.
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
                raise ValueError(
                    "There can only be one dot "
                    "'.' in non-uns representation keys."
                )
            if parts[0] not in ALLOWED_KEYS:
                raise ValueError(
                    f"AnnData annotation key should be one of {ALLOWED_KEYS}."
                )
            if len(parts[1]) >= 120:
                raise ValueError(
                    "Columns keys should be less than 120 characters."
                )
            return val

        @field_validator('*')
        def rep_format_is_correct(cls, val, info):
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
            if not info.field_name.endswith('_key'):
                return val

            match val:
                case str() as v:
                    return cls._validate_single_rep_key(v)
                case [*vals]:
                    return [cls._validate_single_rep_key(v) for v in vals]
                case {**vals}:
                    return {k: cls._validate_single_rep_key(v)  # type: ignore
                            for k, v in vals.items()}
                case None:
                    return None
                case _:
                    raise ValueError(
                        f"Could not interpret format for {info.field_name}. "
                        "Please make sure it is a str, list[str], "
                        "or dict[str, str]."
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
            # if len(upstream_prefix) > 0:  # happens when GroupProcess modules are stacked
            #     upstream_prefix += splitter
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

    def store_item(self, key: str, val: REP, /, add_prefix: bool = True) -> None:
        """Will store the value to a key for lazy saving into adata."""
        if add_prefix:
            key = BaseProcessor.__insert_prefix(key, self.cfg.save_key_prefix)
        self.storage[key] = val

    def store_items(self, items: Dict[str, REP], add_prefix: bool = True):
        """Stores multiple items in dict fashion."""
        if add_prefix:
            items = {
                BaseProcessor.__insert_prefix(
                    key, self.cfg.save_key_prefix): val
                for key, val in items.items()
            }
        self.storage.update(items)

    @staticmethod
    def __insert_prefix(key: str, prefix: str):
        first_key, store_keys = key.split('.', maxsplit=1)
        return f'{first_key}.{prefix}{store_keys}'

    @staticmethod
    def _get_repr(
        adata: AnnData,
        key: str,
        read_key_prefix: str = '',
        to_numpy: bool = False,
    ) -> Any:
        """Get the data representation that key points to."""
        if key is None:
            raise ValueError("Cannot get representation if 'key' is None.")

        if key == 'X':
            return adata.X
        if key in ['obs_names', 'var_names']:
            return getattr(adata, key).to_numpy().astype(str)

        key = BaseProcessor.__insert_prefix(key, read_key_prefix)

        read_class, *read_keys = key.split('.')
        # We only support dictionary style access for read_keys
        rec_itemgetter = compose(*(itemgetter(rk) for rk in read_keys))
        klas = getattr(adata, read_class)
        item = rec_itemgetter(klas)
        if to_numpy:
            item = item.to_numpy()
        return item

    def get_repr(self, adata: AnnData, key: REP_KEY, **kwargs) -> REP:
        """Get the representation(s) that read_key points to."""
        single_get_func = partial(BaseProcessor._get_repr, adata, **kwargs)
        vals: List | Dict
        match key:
            case str() as v:
                return single_get_func(v)
            case [*vals]:
                return [single_get_func(v) for v in vals]
            case {**vals}:
                return {k: single_get_func(v) for k, v in vals.items()}
            case _:
                raise ValueError(f"'{key}' format not understood.")

    @staticmethod
    def _set_repr(adata: AnnData, key: str, value: Any):
        """Save value under the key pointed to by key.
        """
        if key is None:
            raise ValueError("Cannot save representation if 'key' is None.")

        # TODO dont allow X obs_names and var_names
        save_class, *save_keys = key.split('.')
        if save_class != 'uns':
            if len(save_keys) > 1:
                logger.warning(
                    "Found non-'uns' save_class, but more than one save key."
                    "Replacing 'save_keys' dots with dashes."
                )
                save_keys = [''.join(save_keys).replace('.', '-')]

        klas = getattr(adata, save_class)
        # Iterate over all save keys and initialize empty dictionaries if
        # the keys are not found.
        zl = ValueError("Found zero-length save key.")
        while len(save_keys) > 1:
            save_key = save_keys.pop(0)
            if len(save_key) < 1:
                raise zl
            if save_key not in klas:
                klas[save_key] = {}
            klas = klas[save_key]
        # Final key
        save_key = save_keys.pop(0)
        if len(save_key) < 1:
            raise zl
        assert len(save_keys) == 0
        # This is in case save_key points to a dictionary already
        if (save_key in klas
            and isinstance(klas[save_key], dict)
            and isinstance(value, dict)
                and len(klas[save_key])):
            klas[save_key] |= value
        else:
            klas[save_key] = value

    def set_repr(self, adata: AnnData, key: REP_KEY, value: REP) -> None:
        """Saves values under the key that save_key points to. Not to be
        called by any derived classes."""
        single_set_func = partial(BaseProcessor._set_repr, adata)
        keys: List | Dict
        vals: List | Dict

        match key, value:
            # Match a string key and Any value
            case str() as key, val:
                single_set_func(key, val)
            # Match a list of keys and a list of vals
            case [*keys], [*vals]:
                if len(keys) != len(vals):
                    raise ValueError(
                        "Inconsistent length between save_key and value."
                    )
                _ = list(starmap(single_set_func, zip(keys, vals)))
            # Match a dict of keys and a dict of vals
            case {**keys}, {**vals}:
                # Make sure all keys exist
                keys_not_found = set(keys).difference(vals)
                if len(keys_not_found) > 0:
                    raise ValueError(
                        f"Keys {keys_not_found} were not found "
                        "in the output dictionary."
                    )
                _ = list(starmap(
                    single_set_func,
                        ((v, vals[k]) for k, v in keys.items())
                ))
            # No match
            case _:
                raise ValueError(
                    "Inconsistent format between value and key. "
                    f"Key has type {type(key)} but value "
                    f"has type {type(value)}."
                )
