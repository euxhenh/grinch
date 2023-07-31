import inspect
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generic, Type, TypeVar

import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, PrivateAttr, field_validator

BaseConfigurableT = TypeVar("BaseConfigurableT", bound="BaseConfigurable")


class _BaseConfigurable:
    """A base meta class for configurable classes. Each subclass C will
    inherit a Config inner class that knows C's type. The type is set upon
    class definition via this meta class. This allows the construction of
    an instance of C by calling C.Config().create() thus allowing
    reproducibility of the object given its Config only.
    """
    class Config(BaseModel, Generic[BaseConfigurableT]):
        """A base config class for initializing configurable objects.
        """
        model_config = {
            'arbitrary_types_allowed': True,
            'validate_assignment': True,
            'extra': 'forbid',
            'validate_default': True,
        }

        _init_cls: Type[BaseConfigurableT] = PrivateAttr()

        def create(self, *args, **kwargs) -> BaseConfigurableT:
            """Initialize and return an object of type `self._init_cls`.
            """
            return self._init_cls(self, *args, **kwargs)

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # Get bases and check if types are valid
        self_class, immediate_super_class = inspect.getmro(cls)[0:2]
        if self_class.__qualname__ != cls.__qualname__:
            raise TypeError(
                f"Expected class {cls.__qualname__}, but found "
                f"{self_class.__qualname__} as the first base. Overriding "
                "class construction mechanism is not allowed."
            )

        # If a class wants to inherit from another `Configurable`, that
        # `Configurable` must be the immediate super class.
        if not issubclass(immediate_super_class, _BaseConfigurable):
            raise TypeError(
                f"Found {immediate_super_class.__qualname__} as an "
                "immediate parent class which is not a Configurable. "
                "Make sure the Configurable parent is first in parent "
                "resolution order."
            )

        # Make sure the cfg argument is what we expect.
        if cls.Config.__qualname__ != f'{cls.__qualname__}.Config':
            raise NotImplementedError(
                f"Class {cls.__qualname__} does not implement a nested "
                "Config class. Instead, found upstream Config named "
                f"{cls.Config.__qualname__}."
            )

        # We require cfg to be type hinted and positional only.
        signature = inspect.signature(cls.__init__)

        _, cfg_parameter = next(islice(signature.parameters.items(), 1, 2))
        if not issubclass(cls.Config, cfg_parameter.annotation):
            raise TypeError(
                f"Config argument to {cls.__qualname__}.__init__ should "
                f"be of explicit type {cls.Config.__name__}."
            )
        if cfg_parameter.kind != cfg_parameter.POSITIONAL_ONLY:
            raise ValueError(
                f"Config argument to {cls.__qualname__}.__init__ should "
                "be positional only. Consider adding / after the argument."
            )

        # Store outer class type in `Config`
        cls.Config._init_cls = cls  # type: ignore

    def __init__(self, cfg: Config, /):
        self.cfg = cfg


class BaseConfigurable(_BaseConfigurable):

    class Config(_BaseConfigurable.Config):

        if TYPE_CHECKING:
            create: Callable[..., 'BaseConfigurable']

        seed: int | None = None
        logs_path: Path = Path('./grinch_logs')  # Default
        sanity_check: bool = Field(False, exclude=True)
        interactive: bool = Field(False, exclude=True)

        @field_validator('logs_path', mode='before')
        def convert_to_Path(cls, val):
            return Path(val)

    cfg: Config

    @property
    def logs_path(self) -> Path:
        return self.cfg.logs_path

    @contextmanager
    def interactive(self, save_path: str | Path | None = None, **kwargs):
        plt.ion()
        yield None
        plt.ioff()

        if save_path is not None:
            self.logs_path.mkdir(parents=True, exist_ok=True)
            # Set good defaults
            kwargs.setdefault('dpi', 300)
            kwargs.setdefault('bbox_inches', 'tight')
            kwargs.setdefault('transparent', True)
            plt.savefig(self.logs_path / save_path, **kwargs)

        plt.clf()
        plt.close()
