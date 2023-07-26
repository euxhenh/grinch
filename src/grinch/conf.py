import abc
import inspect
from itertools import islice
from typing import ClassVar, List, Tuple

from pydantic import BaseModel, Field

from .reporter import Report, Reporter

reporter = Reporter()


class _BaseConfigurable(abc.ABC):
    """A base class for configurable classes.
    """
    class Config:
        ...

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
        cls.Config.init_type = cls  # type: ignore


class BaseConfig(BaseModel):

    model_config = {
        'arbitrary_types_allowed': True,
        'validate_assignment': True,
        'extra': 'forbid',
        'validate_default': True,
    }

    @property
    def init_type(self):
        """Return the type of the object that this class belongs to. Useful
        when trying to call static or class methods of the underlying type.
        """
        if hasattr(self, '__init_type'):
            return self.__init_type
        return None

    @init_type.setter
    def init_type(self, value):
        self.__init_type = value

    def initialize(self, *args, **kwargs):
        """Initialize and return an object of type `self.init_type`.
        """
        initialized_obj: BaseConfigurable | None = None
        if self.init_type is not None:
            initialized_obj = self.init_type(self, *args, **kwargs)
        return initialized_obj


class BaseConfigurable(_BaseConfigurable):

    class Config(BaseConfig):
        seed: int | None = None
        sanity_check: ClassVar[bool] = Field(False)

    cfg: Config

    def __init__(self, cfg: Config, /):
        self.cfg = cfg
        self._reporter = reporter

    def log(
        self,
        message: str,
        shape: Tuple[int, int] | None = None,
        artifacts: str | List[str] | None = None
    ) -> None:
        """Sends a report to reporter for logging.

        Parameters
        __________
        message: str
        shape: tuple[int, int]
            Shape of the anndata at the point of logging.
        artifacts: str or list of str
            If any artifacts were saved (e.g., images), passing the
            filepath(s) here will log them along with the message.
        """
        report = Report(
            cls=self.__class__.__name__,
            config=self.cfg.model_dump(),
            message=message,
            shape=shape,
            artifacts=artifacts,
        )
        self._reporter.log(report)
