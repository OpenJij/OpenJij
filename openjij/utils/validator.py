# Copyright 2023 Jij Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import typing as typ
from collections.abc import Mapping, Sequence


def raise_type_error(var_name: str, obj: typ.Any, type_: typ.Any) -> None:
    """Raise TypeError.

    Args:
        obj (Any): target object
        type_ (Any): target type
    """
    if is_roughly_correct_type(obj, type_):
        return
    raise TypeError(f"Type of `{var_name}` is {_type_name(obj)}, but {type_} is expected.")


def is_roughly_correct_type(obj: typ.Any, type_: typ.Any) -> bool:
    """Check if obj is roughly correct type.

    The roughly correct type means that the first element of the sequence and mapping only checked.

    Args:
        obj (Any): target object
        type_ (Any): target type
    
    Returns:
        bool: True if obj is roughly correct type, otherwise False
    """

    origin_type = typ.get_origin(type_)

    if origin_type is typ.Union:
        union_types = typ.get_args(type_)
        return any(is_roughly_correct_type(obj, union_type) for union_type in union_types)
    elif origin_type is typ.Optional:
        optional_type = typ.get_args(type_)[0]
        if obj is None:
            return True
        return is_roughly_correct_type(obj, optional_type)

    if origin_type is None:
        origin_type = type_
    if not isinstance(obj, origin_type):
        return False
    
    if isinstance(obj, Mapping):
        if len(obj) == 0:
            return True
        value = next(iter(obj.values()))
        key_type, value_type = typ.get_args(type_)

        value_check = is_roughly_correct_type(value, value_type)
        key_check = is_roughly_correct_type(next(iter(obj.keys())), key_type)
        return value_check and key_check

    if isinstance(obj, Sequence) and not isinstance(obj, str):
        if not obj:
            return True

        item_type = typ.get_args(type_)
        return is_roughly_correct_type(obj[0], item_type)

    return True


def _type_name(obj: typ.Any) -> str:
    """Get type name of obj.

    Args:
        obj (Any): target object
    
    Returns:
        str: type name of obj
    """
    if isinstance(obj, Sequence) and not isinstance(obj, str):
        sequence_name = type(obj).__name__
        if len(obj) == 0:
            return f"{sequence_name}[Any]"
        return f"{sequence_name}[{_type_name(obj[0])}]"
    if isinstance(obj, Mapping):
        map_name = type(obj).__name__
        if len(obj) == 0:
            return f"{map_name}[Any, Any]"
        return f"{map_name}[{_type_name(next(iter(obj.keys())))}, {_type_name(next(iter(obj.values())))}]"
    return type(obj).__name__
