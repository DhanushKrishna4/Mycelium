"""Original, reproducibly drawn premultiplied RGBA targets (no training dependency)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import zlib

import numpy as np

TARGET_NAMES = ("ring", "leaf", "rosette")
TARGET_VERSION = 2


def premultiply(rgba: np.ndarray) -> np.ndarray:
    """Convert straight RGBA in [0, 1] to float32 premultiplied RGBA."""
    result = np.asarray(rgba, dtype=np.float32).copy()
    if result.ndim != 3 or result.shape[-1] != 4:
        raise ValueError("target must be H×W×4 RGBA")
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("target components must be finite and in [0, 1]")
    result[..., :3] *= result[..., 3:4]
    return result


def target_hash(target: np.ndarray) -> str:
    """SHA-256 of contiguous little-endian float32 HWC target data."""
    return hashlib.sha256(np.asarray(target, dtype="<f4").tobytes(order="C")).hexdigest()


def generate_target(name: str, size: int = 32) -> np.ndarray:
    """Draw on a 4× supersampled grid, then area-average premultiplied RGBA.

    Cell (16, 16) is the geometric center, matching the seed convention. The
    transition has no access to these drawing coordinates or target arrays.
    """
    if name not in TARGET_NAMES:
        raise ValueError(f"unknown target {name!r}; choose {', '.join(TARGET_NAMES)}")
    if size != 32:
        raise ValueError("release artwork is defined on a 32×32 canvas")
    sampling = 4
    axis = (np.arange(size * sampling, dtype=np.float64) + .5) / sampling - .5 - size // 2
    x, y = np.meshgrid(axis, axis)
    r = np.sqrt(x*x + y*y)
    rgba = np.zeros((size*sampling, size*sampling, 4), dtype=np.float64)
    if name == "ring":
        # An opaque dark core keeps the initial seed on target tissue. The v1
        # transparent hole caused absorbing extinction in the measured pilot.
        shape = r <= 10
        color = (0.36, 0.77, 0.67)
        rgba[..., :3] = color
        rgba[r < 4, :3] = (0.12, 0.22, 0.20)
    elif name == "leaf":
        u = (x-y) / np.sqrt(2)
        v = (x+y) / np.sqrt(2)
        shape = (u / 10.6)**2 + (v / 6.2)**2 <= 1
        rgba[..., :3] = (0.66, 0.80, 0.34)
        vein = np.abs(v) < .55
        rgba[vein, :3] = (0.83, 0.91, 0.54)
    else:
        theta = np.arctan2(y, x)
        shape = r <= 8.6 + 1.7*np.cos(6*theta)
        rgba[..., :3] = (0.87, 0.57, 0.48)
        rgba[r < 3.1, :3] = (0.97, 0.80, 0.43)
    rgba[..., 3] = shape
    rgba[..., :3] *= rgba[..., 3:4]
    return rgba.reshape(size, sampling, size, sampling, 4).mean(axis=(1, 3)).astype(np.float32)


def write_png(path: str | Path, premultiplied_rgba: np.ndarray, scale: int = 1) -> None:
    """Write a straight-alpha PNG; unpremultiply only the output copy."""
    pixels = np.asarray(premultiplied_rgba, dtype=np.float32).copy()
    if pixels.ndim != 3 or pixels.shape[-1] != 4 or not np.isfinite(pixels).all():
        raise ValueError("PNG input must be finite H×W×4")
    alpha = np.clip(pixels[..., 3:4], 0, 1)
    rgb = np.divide(pixels[..., :3], alpha, out=np.zeros_like(pixels[..., :3]), where=alpha > 0)
    pixels = (np.clip(np.concatenate((rgb, alpha), axis=-1), 0, 1)*255 + .5).astype(np.uint8)
    if scale != 1:
        if not isinstance(scale, int) or not 1 <= scale <= 32:
            raise ValueError("PNG scale must be an integer from 1 to 32")
        pixels = np.repeat(np.repeat(pixels, scale, axis=0), scale, axis=1)
    height, width = pixels.shape[:2]
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind+data) & 0xffffffff)
    encoded = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    encoded += chunk(b"IDAT", zlib.compress(b"".join(b"\0" + row.tobytes() for row in pixels), 9))
    encoded += chunk(b"IEND", b"")
    Path(path).write_bytes(encoded)


def write_targets(directory: str | Path = "targets") -> list[dict]:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for name in TARGET_NAMES:
        target = generate_target(name)
        occupied = np.argwhere(target[..., 3] > 0)
        record = {"name": name, "version": TARGET_VERSION, "shape": list(target.shape),
                  "dtype": "float32", "layout": "HWC", "alpha": "premultiplied",
                  "sha256": target_hash(target), "hash_encoding": "little-endian-float32-HWC",
                  "preprocessing": "4x supersampling; premultiplied area average",
                  "bounds_xyxy": [int(occupied[:, 1].min()), int(occupied[:, 0].min()),
                                  int(occupied[:, 1].max()+1), int(occupied[:, 0].max()+1)],
                  "license": "MIT; original procedural artwork"}
        np.save(destination / f"{name}.npy", target, allow_pickle=False)
        write_png(destination / f"{name}.png", target)
        (destination / f"{name}.json").write_text(json.dumps(record, indent=2)+"\n")
        records.append(record)
    return records


def load_target(name_or_path: str) -> np.ndarray:
    if name_or_path in TARGET_NAMES:
        return generate_target(name_or_path)
    path = Path(name_or_path)
    if path.suffix.lower() != ".png":
        raise ValueError("custom targets must be local PNG files")
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        if source.width*source.height > 16_000_000:
            raise ValueError("custom PNG exceeds the 16 megapixel input limit")
        image = ImageOps.contain(source.convert("RGBA"), (22, 22), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (32, 32))
    canvas.paste(image, ((32-image.width)//2, (32-image.height)//2))
    return premultiply(np.asarray(canvas, dtype=np.float32)/255)
