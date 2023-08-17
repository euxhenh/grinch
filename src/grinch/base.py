import inspect
import logging
from itertools import islice
from operator import itemgetter
from typing import Any, Callable, Dict

import numpy as np
from anndata import AnnData
from pydantic import validate_call

from .utils.ops import compose, safe_format

logger = logging.getLogger(__name__)


class StorageMixin:
    r"""Mixin class for classes that read and store data into AnnData.

    Supports reading and writing values into adata using a
    'column.key_name' format. E.g., `self.read(adata, 'obs.leiden')` will read
    `adata.obs['leiden']` and `self.read(adata, 'X)` will read `adata.X`.
    The benefit of using this class comes from simplifying access for
    composite keys such as those stored under the `uns` column. E.g., the
    following is valid: `self.read(adata, 'uns.group-main.leiden-0.ttest')`
    and will map to `adata.uns['group-main']['leiden-0']['ttest']`. Same
    logic applies to writing objects.

    In addition, this class handles prefixes for all keys. This can be used
    by classes such as GroupBy to set a prefix for all downstream keys.
    E.g., if running a t-Test using 'leiden' clusters as groups, one can set
    the prefix for all those t-Tests to be 'TTest-leiden.' followed by the
    cluster ID itself.

    Finally, this class provides a storage dictionary which holds keys
    *to-be-written*. This is useful when the processor wants to hold writing
    save keys until the end (e.g., when a processor runs on a view of
    adata, its caller (having access to the full adata) may be the one that
    actually wants to do the writing by reading `self.storage`.)

    Attributes
    ----------
    __columns__ : List[str]
        List of allowed AnnData columns.

    prefix : str
        The prefix to prepend to all save keys after the column.

    storage : Dict[str, Any]
        A dict mapping a key to a representation.
    """
    __columns__ = ['obs', 'var', 'obsm', 'varm', 'obsp', 'varp', 'uns']

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

        The storage dict will be used by the `lazy_writer` decorator to
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

        Parameters
        ----------
        key : str
            The key to use for writing. Can contain dots '.'.
        val : Any
            The object to store.
        add_prefix : bool, default=True.
            Whether to prepend `self.prefix` to the key, after the column.
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

    def update_prefix(self, suffix: str, **kwargs) -> None:
        """Appends suffix to prefix.

        This is useful for example, with recursive `GroupProcessor`'s,
        where each `GroupProcessor` needs to prepend some string to its
        `save_key`'s.

        Parameters
        ----------
        suffix : str
            The key to append (will be formatted if it has curly brackets).
        """
        self.prefix += safe_format(suffix, **kwargs)

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

        Parameters
        ----------
        adata : AnnData
            The annotated data structure.
        key : str
            The key to read from.
        prefix : str
            If non-empty, will prepend to the key after the column.
        to_numpy : bool, default=False
            If True, will call `to_numpy()` on the retrieved object.

        Returns
        -------
        item: Any
            The object read from adata columns. This is likely going to be
            a view so care should be taken when updating it inplace.
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
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def write(adata: AnnData, key: str, value: Any) -> None:
        """Immediately save value under the key pointed to by key.

        Keys should contain a dot '.' that splits the anndata column with
        the key name. E.g., `obs.leiden` will write into
        `adata.obs['leiden']`. If the column is not `uns` and key contains
        more than one dot, all dots after the first will be replaced with
        dashes. Multiple dots are allowed when storing into `uns`. Any
        non-existing intermediate dictionaries will be created on the fly.

        Parameters
        ----------
        adata : AnnData
            The annotated data structure.
        key : str
            The key to store `value` into.
        value : Any
            The object to store under `key`.
        """
        save_class, *save_keys = key.split('.')

        if save_class not in (cols := StorageMixin.__columns__):
            raise ValueError(f"Save class not in allowed list={cols}.")
        # Make sure save keys are not empty
        for _ in filter(lambda x: len(x) == 0, save_keys):
            raise ValueError("Found 'save_key' of zero-length.")

        # Make sure we have no dots in non-uns columns.
        if save_class != 'uns':
            if len(save_keys) > 1:
                logger.warning(
                    "Found non-'uns' save_class, but more than one save key."
                    "Replacing 'save_keys' dots with dashes."
                )
                # Only uns allows nested dict's
                save_keys = [''.join(save_keys).replace('.', '-')]

        klas = getattr(adata, save_class)
        # Iterate over all save keys and initialize empty dictionaries if
        # the keys are not found.
        while len(save_keys) > 1:
            save_key = save_keys.pop(0)
            if save_key not in klas:
                klas[save_key] = {}
            klas = klas[save_key]

        save_key = save_keys.pop(0)  # Last key
        # If save_key points to an existing dictionary, we merge the results.
        if save_key in klas and len(klas[save_key]):
            if isinstance(klas[save_key], dict) and isinstance(value, dict):
                klas[save_key] |= value
                return
        klas[save_key] = value

    @staticmethod
    def lazy_writer(f: Callable, /):
        """A decorator for functions that wish to lazily store items.

        Will reset `self.storage` on function call. To update this, the
        caller should use the `store_item` and `store_items` methods.
        """
        params = inspect.signature(f).parameters
        if len(params) < 2:  # require 'self' and 'adata'
            raise ValueError("A 'setter_method' should take at least 2 arguments.")

        _, adata_parameter = next(islice(params.items(), 1, 2))
        if not issubclass(AnnData, adata_parameter.annotation):
            raise ValueError(
                "First argument to a 'setter_method' "
                "should be explicitly typed 'AnnData'."
            )

        def _wrapper(self: StorageMixin, adata: AnnData,
                     *args, **kwargs) -> Dict[str, Any] | None:
            """Call f and either write or return storage."""
            self.storage.clear()  # Empty storage dict before function call
            if f(self, adata, *args, **kwargs) is not None:  # Also calls f
                logger.warning(f"Function '{f.__name__}' returned a value.")

            if kwargs.get('return_storage', False):
                return self.storage
            for key, item in self.storage.items():
                self.write(adata, key, item)
            return None

        return _wrapper
