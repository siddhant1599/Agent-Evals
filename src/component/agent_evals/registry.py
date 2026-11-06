"""The six registries.

One generic Registry class, six instances -- not six copy-pasted dicts. Adding a variant means
subclassing a base and registering it by name; configs then select by string and no
orchestration code imports a concrete implementation.

A registry only contains what has been imported. If nobody imports ``topologies/t3_debate.py``,
T3 silently does not exist and a config naming it fails at build time rather than import time.
Each subpackage's ``__init__`` must therefore import its implementations explicitly, and
``src/tests/test_registry.py`` asserts the expected names are present.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Generic, TypeVar

__all__ = [
    "Registry",
    "RegistryError",
    "SERVING",
    "DATASETS",
    "AGENTS",
    "TOPOLOGIES",
    "METRICS",
    "REWARDS",
    "ALL_REGISTRIES",
]

T = TypeVar("T")


class RegistryError(KeyError):
    """Raised for an unknown or duplicate registry name."""


class Registry(Generic[T]):
    """A name -> class lookup for one extension point.

    Stores classes, not instances or functions. ``build`` constructs; ``get`` returns the class
    for callers that need to inspect it without constructing.

    Args:
        kind: What this registry holds, e.g. "topology". Used in error messages only.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, type[T]] = {}

    @property
    def kind(self) -> str:
        """What this registry holds, for error messages."""
        return self._kind

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Return a decorator registering a class under ``name``.

        Args:
            name: The string a config will use to select this implementation.

        Returns:
            A decorator that registers the class and returns it unchanged.

        Raises:
            RegistryError: If ``name`` is already registered. Two implementations answering to
                one name is always a mistake, and silently keeping the last one imported makes
                it depend on import order.
        """

        def decorator(cls: type[T]) -> type[T]:
            if name in self._entries:
                existing = self._entries[name].__name__
                raise RegistryError(
                    f"{self._kind} {name!r} is already registered to {existing}; "
                    f"cannot also register {cls.__name__}"
                )
            self._entries[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        """Return the class registered under ``name``.

        Args:
            name: The registered name.

        Returns:
            The registered class.

        Raises:
            RegistryError: If nothing is registered under ``name``. The message lists every
                available name, because the usual cause is a typo or a missing import.
        """
        try:
            return self._entries[name]
        except KeyError:
            available = ", ".join(self.names()) or "<nothing registered>"
            raise RegistryError(
                f"unknown {self._kind} {name!r}. Available: {available}. "
                f"If the name looks right, the module defining it was probably never imported."
            ) from None

    def build(self, name: str, *args: object, **kwargs: object) -> T:
        """Construct the implementation registered under ``name``.

        Args:
            name: The registered name.
            *args: Positional arguments forwarded to the constructor.
            **kwargs: Keyword arguments forwarded to the constructor.

        Returns:
            A new instance.

        Raises:
            RegistryError: If nothing is registered under ``name``.
        """
        return self.get(name)(*args, **kwargs)

    @contextmanager
    def temporary(self, name: str, cls: type[T]) -> Iterator[type[T]]:
        """Register ``cls`` for the duration of a ``with`` block, then remove it.

        Exists so a test can exercise registry-driven code against a stub without leaving the
        stub in a process-wide registry. Nothing in the package should register temporarily:
        a real implementation registers at import time and stays.

        Args:
            name: The name to register under, which must be free.
            cls: The class to register.

        Yields:
            ``cls`` unchanged.

        Raises:
            RegistryError: If ``name`` is already registered.
        """
        self.register(name)(cls)
        try:
            yield cls
        finally:
            self._entries.pop(name, None)

    def names(self) -> tuple[str, ...]:
        """Every registered name, sorted."""
        return tuple(sorted(self._entries))

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry(kind={self._kind!r}, names={self.names()})"


#: Backbone / serving adapters.
SERVING: Registry[object] = Registry("serving adapter")

#: Tier 1/2/3 dataset adapters.
DATASETS: Registry[object] = Registry("dataset adapter")

#: Agent roles: generator, critic, planner, executor, verifier, judge.
AGENTS: Registry[object] = Registry("agent role")

#: Topologies T0 through T5.
TOPOLOGIES: Registry[object] = Registry("topology")

#: Factuality, NLG quality and citation metrics.
METRICS: Registry[object] = Registry("metric")

#: GRPO reward functions.
REWARDS: Registry[object] = Registry("reward")

#: Every registry, for diagnostics and for the registration-completeness test.
ALL_REGISTRIES: tuple[Registry[object], ...] = (
    SERVING,
    DATASETS,
    AGENTS,
    TOPOLOGIES,
    METRICS,
    REWARDS,
)
