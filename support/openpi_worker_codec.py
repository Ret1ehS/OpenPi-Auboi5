from __future__ import annotations

from typing import Any

import numpy as np

_NDARRAY_MARKER = "__openpi_worker_ndarray__"
_NUMPY_SCALAR_MARKER = "__openpi_worker_numpy_scalar__"


def encode_worker_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            _NDARRAY_MARKER: True,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": value.tobytes(),
        }
    if isinstance(value, np.generic):
        return {
            _NUMPY_SCALAR_MARKER: True,
            "dtype": value.dtype.str,
            "value": value.item(),
        }
    if isinstance(value, dict):
        return {key: encode_worker_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [encode_worker_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(encode_worker_value(item) for item in value)
    return value


def decode_worker_value(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get(_NDARRAY_MARKER):
            array = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
            return array.reshape(value["shape"]).copy()
        if value.get(_NUMPY_SCALAR_MARKER):
            return np.asarray(value["value"], dtype=np.dtype(value["dtype"]))[()]
        return {key: decode_worker_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_worker_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(decode_worker_value(item) for item in value)
    return value


__all__ = ["encode_worker_value", "decode_worker_value"]
