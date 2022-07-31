import abc
import logging
from itertools import starmap
from functools import partial
from typing import Any, Dict, List, Optional

from anndata import AnnData
from pydantic import validate_arguments, validator

from .aliases import ALLOWED_KEYS, REP_KEY
from .conf import BaseConfigurable

logger = logging.getLogger(__name__)


class BaseProcessor(BaseConfigurable):
    """A base class for dimensionality reduction, clustering and other
    processors. A processor cannot update the data matrix X, but can use it
    to perform any kind of fitting.
    """

    class Config(BaseConfigurable.Config):
        inplace: bool = True
        # Select the representation to use. If str: if 'X', will use
        # adata.X, otherwise it must contain a dot that splits the
        # annotation key that will be used and the column key. E.g.,
        # 'obsm.x_emb' will use 'adata.obsm['x_emb']'. A list of str will
        # be parsed as *args, and a dict of (str, str) should map a
        # dictionary key to the desired representation. The latter is
        # useful when calling, for example, predictors which require a data
        # representation X and labels y. In this case, X and y would be
        # dictionary keys and the corresponding representations for X and y
        # would be the values.
        read_key: REP_KEY = None
        save_key: REP_KEY = None

        @staticmethod
        def _validate_single_rep_key(val: str, allow_x: bool = True):
            """Validates the format of a single key (str)."""
            if val is None or (val == 'X' and allow_x):
                return val
            elif val == 'X':
                raise ValueError(
                    "'save_key' rep cannot equal X. Maybe you meant to use a transform instead."
                )
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

        @validator('read_key', 'save_key')
        def rep_format_is_correct(cls, val, field):
            allow_x = field.name != 'save_key'
            match val:
                case str() as v:
                    return cls._validate_single_rep_key(v, allow_x)
                case [*vals]:
                    return [cls._validate_single_rep_key(v, allow_x) for v in vals]
                case {**vals}:
                    return {k: cls._validate_single_rep_key(v, allow_x) for k, v in vals.items()}
                case _:
                    raise ValueError(
                        f"Could not interpret format for {field.name}. Please make sure "
                        "it is a str, list[str], or dict[str, str]."
                    )

    cfg: Config

    @property
    def processor(self):
        """Points to the object that is being wrapped by the derived class."""
        return getattr(self, '_processor', None)

    @processor.setter
    def processor(self, value):
        self._processor = value

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __call__(self, adata: AnnData) -> Optional[AnnData]:
        if not self.cfg.inplace:
            adata = adata.copy()

        self._process(adata)

        return adata if not self.cfg.inplace else None

    @abc.abstractmethod
    def _process(self, adata: AnnData) -> None:
        """To be implemented by a derived class."""
        raise NotImplementedError

    @staticmethod
    def _get_single_repr(adata: AnnData, key: str) -> Any:
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

    def _get_repr(self, adata: AnnData) -> Any | List[Any] | Dict[str, Any]:
        """Get the representation(s) that read_key points to."""
        match self.cfg.read_key:
            case str() as v:
                return self._get_single_repr(adata, v)
            case [*vals]:
                return [self._get_single_repr(adata, v) for v in vals]
            case {**vals}:
                return {k: self._get_single_repr(adata, v) for k, v in vals.items()}

    @staticmethod
    def _set_single_repr(adata: AnnData, key: str, value: Any, cfg: Optional[Config] = None):
        """Save value under the key pointed to by key. Also saves
        config under `uns` if `save_config` is True.
        """
        if key is None:
            raise ValueError("Cannot save representation if 'save_key' is None.")

        save_class, save_key = key.split('.')
        try:
            getattr(adata, save_class)[save_key] = value
        except Exception:
            # Try initializing to an empty dictionary on fail
            setattr(adata, save_class, {})
            getattr(adata, save_class)[save_key] = value

        if cfg:
            adata.uns[save_key] = cfg

    def _set_repr(
        self,
        adata: AnnData,
        value: Any | List[Any] | Dict[str, Any],
        save_config: bool = True
    ) -> None:
        set_func = partial(self._set_single_repr, adata)
        if save_config:
            set_func = partial(set_func, cfg=self.cfg.dict())

        """Saves values under the key that save_key points to."""
        match self.cfg.save_key, value:
            # Match a string key and Any value
            case str() as key, val:
                set_func(key, val)
            # Match a list of keys and a list of vals
            case [*keys], [*vals]:
                if len(keys) != len(vals):
                    raise ValueError("Inconsistent length between save_key and value.")
                starmap(set_func, zip(keys, vals))
            # Match a dict of keys and a dict of vals
            case {**keys}, {**vals}:
                # Make sure all keys exist
                keys_not_found = set(keys).difference(vals)
                if len(keys_not_found) > 0:
                    raise ValueError(
                        f"Keys {keys_not_found} were not found in the output dictionary."
                    )
                starmap(set_func, ((v, vals[k]) for k, v in keys.items()))
            # No match
            case _:
                raise ValueError("Inconsistent format between value and save_key.")
