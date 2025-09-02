"""Base config class"""
from dataclasses import asdict, dataclass, fields
from typing import Iterator,Tuple, Any

@dataclass
class BaseConfig:
    """Base config class"""
    def __setitem__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"{key} is not a valid field")

    def __getitem__(self, key: str):
        """support args[key]"""
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
        for config_cls in self.__class__.__mro__:
            if issubclass(config_cls, BaseConfig):
                # NOTE: if config_cls does not implement '_validate_impl' but inherits
                # from parent class, it will not show in config_cls.__dict__
                if '_validate_impl' in config_cls.__dict__:
                    config_cls._validate_impl(self)

        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, BaseConfig):
                value.validate()

    def _validate_impl(self):
        """valid this config, recursively called in `validate`.
        Should raise Error if failed.
        """
        return

    def __post_init__(self):
        for config_cls in self.__class__.__mro__:
            if issubclass(config_cls, BaseConfig):
                config_cls._post_init_impl(self)

    def _post_init_impl(self):
        """post init implementation
        """
        return
