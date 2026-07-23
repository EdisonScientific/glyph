from __future__ import annotations

import types
from typing import Any, Union, get_args, get_origin, get_type_hints


def coerce_yaml_values(data: dict[str, Any], cls: type) -> dict[str, Any]:
    """Coerce string YAML values to the numeric types declared on a dataclass.

    PyYAML parses ``learning_rate: 1e-5`` as the string ``'1e-5'`` because the
    YAML 1.1 float regex requires a decimal point. Training code then crashes
    far away from the config with a confusing arithmetic error. This walks each
    loaded value and, when the dataclass annotates the field as numeric, casts
    string values via ``int()`` / ``float()`` / case-insensitive bool. Values
    that fail to coerce are left alone so the dataclass constructor still
    surfaces a clear error.
    """
    hints = get_type_hints(cls)
    out: dict[str, Any] = {}
    for k, v in data.items():
        target = hints.get(k)
        if target is None or not isinstance(v, str):
            out[k] = v
            continue
        if get_origin(target) in (Union, types.UnionType):
            non_none = [a for a in get_args(target) if a is not type(None)]
            if len(non_none) == 1:
                target = non_none[0]
        try:
            if target is float:
                out[k] = float(v)
            elif target is int:
                out[k] = int(v)
            elif target is bool:
                out[k] = v.strip().lower() in ("true", "yes", "1", "on")
            else:
                out[k] = v
        except (ValueError, TypeError):
            out[k] = v
    return out
