import abc
import logging
from functools import partial
from itertools import starmap
from typing import Any, List, Optional

from anndata import AnnData
from pydantic import validate_arguments, validator

from .aliases import ALLOWED_KEYS, REP, REP_KEY
from .conf import BaseConfigurable
from .utils.validation import check_has_processor

logger = logging.getLogger(__name__)


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
    *_key: aliases.REP_KEY
        Any Config member parameter that ends in '_key' will be checked by
        pydantic validators to conform with adata column names.
    """

    class Config(BaseConfigurable.Config):
        inplace: bool = True

        @staticmethod
        def _validate_single_rep_key(val: str):
            """Validates the format of a single key (str)."""
            if val is None or val == 'X':
                return val
            if '.' not in val:
                raise ValueError(
                    "Representation keys must equal 'X' or must contain a "
                    "dot '.' that points to the AnnData column to use."
                )
            if len(parts := val.split('.')) > 2:
                # TODO Allow more dots for uns dictionaries.
                raise ValueError("There can only be one dot '.' in representation keys.")
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

    cfg: Config

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
        """Upon assignment, will check if the processor contains the field
        listed in this list.
        """
        return []

    @staticmethod
    def _processor_stats() -> List[str]:
        """Processor attributes to extract into a stats dictionary for writing.
        """
        return []

    def save_processor_stats(self, adata: AnnData) -> None:
        check_has_processor(self)
        if not hasattr(self.cfg, 'stats_key'):
            raise KeyError(f"No 'stats_key' was found in {self.cfg.__qualname__}.")
        # Assume it has been explicitly set to None
        if self.cfg.stats_key is None:
            return

        stats = {stat: getattr(self.processor, stat) for stat in self._processor_stats()}
        if stats:
            self.set_repr(adata, self.cfg.stats_key, stats)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __call__(self, adata: AnnData) -> Optional[AnnData]:
        """Calls the processor with adata. Will copy adata if inplace was
        set to False.
        """
        if not self.cfg.inplace:
            adata = adata.copy()

        self._process(adata)

        return adata if not self.cfg.inplace else None

    @abc.abstractmethod
    def _process(self, adata: AnnData) -> None:
        """To be implemented by a derived class."""
        raise NotImplementedError

    @staticmethod
    def _get_repr_single(adata: AnnData, key: str) -> Any:
        """Get the data representation that key points to."""
        if key is None:
            raise ValueError("Cannot get representation if 'key' is None.")

        if key == 'X':
            return adata.X

        read_class, read_key = key.split('.')
        klas = getattr(adata, read_class)
        if read_key not in klas:
            raise ValueError(f"Could not find {read_key} in adata.{read_class}.")
        return klas[read_key]

    @staticmethod
    def get_repr(adata: AnnData, key: REP_KEY) -> REP:
        """Get the representation(s) that read_key points to."""
        single_get_func = partial(BaseProcessor._get_repr_single, adata)
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
    def _set_repr_single(adata: AnnData, key: str, value: Any):
        """Save value under the key pointed to by key. Also saves
        config under `uns` if `save_config` is True.
        """
        if key is None:
            raise ValueError("Cannot save representation if 'key' is None.")

        save_class, save_key = key.split('.')
        try:
            getattr(adata, save_class)[save_key] = value
        except Exception:
            # Try initializing to an empty dictionary on fail
            setattr(adata, save_class, {})
            getattr(adata, save_class)[save_key] = value

    @staticmethod
    def set_repr(adata: AnnData, key: REP_KEY, value: REP) -> None:
        single_set_func = partial(BaseProcessor._set_repr_single, adata)

        """Saves values under the key that save_key points to."""
        match key, value:
            # Match a string key and Any value
            case str() as key, val:
                single_set_func(key, val)
            # Match a list of keys and a list of vals
            case [*keys], [*vals]:
                if len(keys) != len(vals):
                    raise ValueError("Inconsistent length between save_key and value.")
                starmap(single_set_func, zip(keys, vals))
            # Match a dict of keys and a dict of vals
            case {**keys}, {**vals}:
                # Make sure all keys exist
                keys_not_found = set(keys).difference(vals)
                if len(keys_not_found) > 0:
                    raise ValueError(
                        f"Keys {keys_not_found} were not found in the output dictionary."
                    )
                starmap(single_set_func, ((v, vals[k]) for k, v in keys.items()))
            # No match
            case _:
                raise ValueError(
                    f"Inconsistent format between value and key. Key has type {type(key)} "
                    f"but value has type {type(value)}."
                )
