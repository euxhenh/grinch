import logging
from functools import partial
from itertools import starmap
from operator import itemgetter
from typing import Any, Dict, List

import numpy as np
from anndata import AnnData
from pydantic import validate_call

from .custom_types import REP_KEY
from .utils.ops import compose, safe_format

logger = logging.getLogger(__name__)


class StorageMixin:
    """Mixin class for classes that store data into AnnData."""

    @property
    def prefix(self) -> str:
        """The prefix to prepend to all save keys."""
        if not hasattr(self, '_prefix'):
            self._prefix = ''  # By default no prefix
        return self._prefix

    @prefix.setter
    def prefix(self, val) -> None:
        """Updates save key prefix."""
        self._prefix = val

    @property
    def storage(self) -> Dict[str, Any]:
        """Return a pointer to the inner storage dict.

        The storage dict will be used by the `adata_modifier` decorator to
        dump all the keys with the respective values into the AnnData
        columns. The keys in storage should not include the upstream prefix
        as this will be added here.
        """
        if not hasattr(self, '_storage'):
            self._storage: Dict[str, Any] = {}
        return self._storage

    def store_item(self, key: str, val: Any, /, add_prefix: bool = True) -> None:
        """Will store the value to a key for lazy saving into adata.

        Updates the inner storage only. Does not set the value in AnnData.
        """
        if add_prefix:
            key = StorageMixin.__insert_prefix_after_col(key, self.prefix)
        self.storage[key] = val

    def store_items(self, items: Dict[str, Any], add_prefix: bool = True) -> None:
        """Will store the values to their keys for lazy saving into adata.

        Updates the inner storage only. Does not set the value in AnnData.
        This takes a dictionary as input. To store a single key, use
        `self.store_item`.
        """
        if add_prefix:
            items = {
                StorageMixin.__insert_prefix_after_col(key, self.prefix): val
                for key, val in items.items()
            }
        self.storage.update(items)

    @staticmethod
    def __insert_prefix_after_col(key: str, prefix: str):
        """Inserts prefix to the key after performing a column split.

        Example
        -------
        If key='uns.ttest' and prefix='group-0.', then this returns
        'uns.group-0.ttest'. Note the dot '.' after 'group-0'.
        """
        first_key, store_keys = key.split('.', maxsplit=1)
        return f'{first_key}.{prefix}{store_keys}'

    def insert_prefix(self, current_key: str, **kwargs) -> str:
        """Returns a key with any upstread prefixes prepended.

        This is useful for example, with recursive `GroupProcessor`'s,
        where each `GroupProcessor` needs to prepend some string to its
        `save_key`'s.

        Parameters
        ----------
        current_key : str
            The key the processor wants to use for its `save_key`.

        Returns
        -------
        new_prefix : str
            The current key with `prefix` prepended.
        """
        current_key = safe_format(current_key, **kwargs)
        return f'{self.prefix}{current_key}'

    @staticmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def read(adata: AnnData, key: str, prefix: str = '', to_numpy: bool = False):
        """Get the data representation that key points to.

        'X', 'obs_names', and 'var_names' are special keys and will be
        mapped to adata.X, adata.obs_names, and adata.var_names,
        respectively. All other keys should contain a dot '.' that splits
        the AnnData column and the key. E.g., 'obs.leiden'. We only perform
        dictionary style access here. So 'obs.leiden' will be mapped to
        adata.obs['leiden']. Multiple dots '.' can be used (but these are
        only supported for 'uns' keys). E.g., 'uns.group-0.ttest.leiden-0'
        will be mapped to adata.uns['group-0']['ttest']['leiden-0'].
        """
        if key == 'X':
            return adata.X
        if key in ['obs_names', 'var_names']:
            return np.asarray(getattr(adata, key)).astype(str)

        key = StorageMixin.__insert_prefix_after_col(key, prefix)
        read_class, *read_keys = key.split('.')

        # Compose attribute access for all read_keys in case there was more
        # than one split.
        rec_itemgetter = compose(*(itemgetter(rk) for rk in read_keys))
        # We only support dictionary style access for read_keys
        item = rec_itemgetter(getattr(adata, read_class))
        return item if not to_numpy else item.to_numpy()

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

    def set_repr(self, adata: AnnData, key: REP_KEY, value: Any) -> None:
        """Saves values under the key that save_key points to. Not to be
        called by any derived classes."""
        single_set_func = partial(self._set_repr, adata)
        keys: List
        vals: List

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
            # No match
            case _:
                raise ValueError(
                    "Inconsistent format between value and key. "
                    f"Key has type {type(key)} but value "
                    f"has type {type(value)}."
                )
