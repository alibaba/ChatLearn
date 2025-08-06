"""Base config class"""
from dataclasses import asdict, dataclass, field
from typing import Iterator,Tuple, Any

@dataclass
class BaseConfig:
    """Base config class"""
    _freeze: bool = field(
        default=False, metadata={"help": "If True, this config cannot be modified."}
    )
    def freeze(self):
        self._freeze = True

    def __setattr__(self, key, value):
        if self._freeze:
            raise ValueError("Attempt to modify a frozen config.")
        object.__setattr__(self, key, value)

    def __setitem__(self, key, value):
        if self._freeze:
            raise ValueError("Attempt to modify a frozen config.")
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"{key} is not a valid field")

    def __getitem__(self, key: str):
        """support args[key]"""
        if self._freeze:
            raise ValueError("Attempt to modify a frozen config.")
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """support key in args"""
        return hasattr(self, key)

    def get(self, key: str, default=None):
        """support args.get(key)"""
        return getattr(self, key, default)

    def items(self) -> Iterator[Tuple[str, Any]]:
        """support args.items()"""
        return asdict(self).items()

    def keys(self) -> Iterator[str]:
        """support args.keys()"""
        return asdict(self).keys()

    def values(self) -> Iterator[Any]:
        """support args.values()"""
        return asdict(self).values()

    def validate(self):
        """valid this config with `_validate_impl` implemented by
        each config class.
        """
        for config_cls in self.__class__.__mro__[1:]:
            if issubclass(config_cls, BaseConfig):
                config_cls._validate_impl(self)

    def _validate_impl(self):
        """valid this config, recursively called in `validate`.
        Should raise Error if failed.
        """
        return
