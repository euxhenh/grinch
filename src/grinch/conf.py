import inspect
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generic, List, Type, TypeVar

from pydantic import BaseModel, Field, PrivateAttr

BaseConfigurableT = TypeVar("BaseConfigurableT", bound="BaseConfigurable")


class _BaseConfigurable:
    r"""A base meta class for configurable classes. Each subclass C will
    inherit a Config inner class that knows C's type. The type is set upon
    class definition via this meta class. This allows the construction of
    an instance of C by calling C.Config().create() thus allowing
    reproducibility of the object given its Config only.

    E.g., BaseConfigurable.Config(**kwargs).create() -> BaseConfigurable.
    This is syntactic sugar for
    BaseConfigurable(BaseConfigurable.Config(**kwargs)) ->
    BaseConfigurable.

    Attributes
    ----------
    __buffers__ : List[str]
        List of field names that are part of the Configurable's state.
        These will be dumped when saving the model.

    cfg : Config
        The Base Config for the Configurable.
    """
    __slots__ = ['cfg']

    __buffers__: List[str] = []

    class Config(BaseModel, Generic[BaseConfigurableT]):
        """A stateless base config class for creating configurable objects.

        Attributes
        ----------
        _init_cls : Type
            The type of the outer configurable class.
        """
        model_config = {
            'arbitrary_types_allowed': True,
            'validate_assignment': True,
            'extra': 'forbid',
            'validate_default': True,
            "populate_by_name": True,
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

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.cfg)})"


class BaseConfigurable(_BaseConfigurable):
    """BaseConfigurable class with an inner Config.

    To be inherited by all configurable classes.
    """
    class Config(_BaseConfigurable.Config):
        r"""BaseConfigurable.Config

        Parameters
        ----------
        logs_path : Path, default=Path('./grinch_logs')
            The path to use for saving logs.

        interactive : bool, default=False
            If ``True`` will run in interactive mode. It is the job of that
            configurable class to make use of this.

        seed : int, default=None
            Random state to use for the wrapped model. It is the job of
            that configurable class to make sure the seed is used.

        sanity_check : bool, default=False
            If ``True`` will make a quick run to ensure that everything
            runs without errors.
        """
        if TYPE_CHECKING:
            create: Callable[..., 'BaseConfigurable']

        logs_path: Path = Path('./grinch_logs', exclude=True, repr=False)
        interactive: bool = Field(False, exclude=True, repr=False)
        # The following two are likely to be used by a GRPipeline
        seed: int | None = None
        sanity_check: bool = Field(False, exclude=True, repr=False)

        def model_post_init(self, _) -> None:
            # Create logs directory if it does not exist
            self.logs_path.mkdir(parents=True, exist_ok=True)

    cfg: Config

    @property
    def logs_path(self) -> Path:
        """The path to logs."""
        return self.cfg.logs_path
