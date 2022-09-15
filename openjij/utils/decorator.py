from __future__ import annotations
import functools, warnings


def deprecated_alias(**aliases):
    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            rename_kwargs(f.__name__, kwargs, aliases)
            return f(*args, **kwargs)

        return wrapper

    return deco


def rename_kwargs(func_name, kwargs, aliases):
    for alias, new in aliases.items():
        if alias in kwargs:
            if new in kwargs:
                raise TypeError(f"{func_name} received both {alias} and {new}")
            warnings.warn(f"{alias} is deprecated; use {new}", DeprecationWarning)
            kwargs[new] = kwargs.pop(alias)


def disabled(func):
    def wrapper(*args, **kwargs):
        raise NotImplementedError(f"The function {func.__name__} is disabled.")

    return wrapper
