"""Bounded, data-only model import/export. Never deserializes executable objects."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .config import ARCHITECTURE, MAX_STATE_ABS, RUNTIME_VERSION, WEIGHT_SHAPES

MODEL_SCHEMA = "mycelium.model.v1"
MAX_MODEL_BYTES = 1_000_000


def bounded_json(source, max_bytes=MAX_MODEL_BYTES):
    if isinstance(source, dict):
        try:
            encoded = json.dumps(source, allow_nan=False, separators=(",", ":"))
        except (ValueError, TypeError, RecursionError) as exc:
            raise ValueError("Import must contain finite JSON data only.") from exc
        if len(encoded.encode("utf-8")) > max_bytes:
            raise ValueError(f"Import exceeds the {max_bytes}-byte limit.")
        return source
    if isinstance(source, bytes):
        raw = source
    elif isinstance(source, str) and source.lstrip().startswith(("{", "[")):
        raw = source.encode("utf-8")
    else:
        path = Path(source)
        if path.stat().st_size > max_bytes:
            raise ValueError(f"Import exceeds the {max_bytes}-byte limit.")
        with path.open("rb") as stream:
            raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"Import exceeds the {max_bytes}-byte limit.")
    try:
        data = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {x}")))
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise ValueError("Import is not valid finite JSON.") from exc
    if not isinstance(data, dict):
        raise ValueError("Import must be a JSON object.")
    return data


def checked_array(value, shape, name, *, dtype=np.float32, max_abs=MAX_STATE_ABS):
    """Verify nested dimensions and scalar types before NumPy allocates an array."""
    def inspect(node, dims):
        if dims:
            if not isinstance(node, (list, tuple)) or len(node) != dims[0]:
                raise ValueError(f"{name} must have shape {tuple(shape)}.")
            for child in node:
                inspect(child, dims[1:])
        elif not isinstance(node, (float, int, bool)) or not -max_abs <= node <= max_abs:
            raise ValueError(f"{name} contains an invalid or excessive scalar.")
    if isinstance(value, np.ndarray):
        if value.shape != tuple(shape) or value.dtype.kind not in "bfiu" or not np.isfinite(value).all() or np.max(np.abs(value), initial=0) > max_abs:
            raise ValueError(f"{name} must be a finite bounded array with shape {tuple(shape)}.")
    else:
        inspect(value, tuple(shape))
    result = np.asarray(value, dtype=dtype)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} is not representable in the required dtype.")
    return result.copy()


def validate_weights(weights):
    if not isinstance(weights, dict) or set(weights) != set(WEIGHT_SHAPES):
        raise ValueError("Weights must contain exactly W1, b1, W2, and b2.")
    return {name: checked_array(weights[name], shape, name) for name, shape in WEIGHT_SHAPES.items()}


def weights_hash(weights):
    """SHA-256 over names, shapes and C-order IEEE754 little-endian float32 bytes."""
    digest = hashlib.sha256()
    for name, shape in WEIGHT_SHAPES.items():
        value = np.asarray(weights[name], dtype="<f4")
        if value.shape != shape:
            raise ValueError(f"Invalid shape for {name}.")
        digest.update(f"{name}:{','.join(map(str, shape))}:float32le\n".encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def model_document(weights, metadata=None):
    weights = validate_weights(weights)
    document = {"schema": MODEL_SCHEMA, "runtime_version": RUNTIME_VERSION,
                "architecture": dict(ARCHITECTURE), "dtype": "float32",
                "weights_hash": weights_hash(weights), "metadata": metadata or {},
                "weights": {key: value.tolist() for key, value in weights.items()}}
    bounded_json(document)
    return document


def save_model(path, weights, metadata=None):
    document = model_document(weights, metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, allow_nan=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)
    return document


def load_model(source):
    document = bounded_json(source)
    if document.get("schema") != MODEL_SCHEMA:
        raise ValueError(f"Unsupported model schema; expected {MODEL_SCHEMA}.")
    if document.get("runtime_version") != RUNTIME_VERSION:
        raise ValueError("Incompatible model runtime version.")
    if document.get("architecture") != ARCHITECTURE or document.get("dtype") != "float32":
        raise ValueError("Incompatible architecture: expected 16 channels, 48→64→16, zero padding, and Mycelium v1 conventions.")
    weights = validate_weights(document.get("weights"))
    actual_hash = weights_hash(weights)
    if document.get("weights_hash") != actual_hash:
        raise ValueError("Model weights hash mismatch; the file may be corrupted.")
    if not isinstance(document.get("metadata", {}), dict):
        raise ValueError("Model metadata must be an object.")
    return {**document, "weights": weights}
