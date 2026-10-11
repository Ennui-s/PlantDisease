"""Validate an uploaded image and turn it into the model's input.

preprocess() reproduces the training pipeline exactly (RGB, bilinear resize to size x size,
scale to [0, 1], no normalisation: torchvision Resize + ToTensor as in data/dataloader.py), using
Pillow and numpy only. `python -m serving.check_parity` verifies this against the training transform.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from serving.errors import ApiError
from serving.settings import Settings


def load_image(data: bytes, settings: Settings) -> Image.Image:
    """Return a decoded RGB image, or raise ApiError describing what is wrong with the upload."""
    if not data:
        raise ApiError(400, "empty_file", "The uploaded file is empty.")
    try:
        image = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError:
        raise ApiError(413, "image_too_large", "The image has too many pixels.") from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise ApiError(415, "unsupported_media_type",
                       "The file is not a readable image in a supported format (JPEG or PNG).") from None

    if image.format not in settings.allowed_formats:
        raise ApiError(415, "unsupported_media_type",
                       f"Unsupported image format {image.format!r}. Upload a JPEG or PNG.")
    width, height = image.size
    if width * height > settings.max_pixels:
        raise ApiError(413, "image_too_large", "The image has too many pixels.")
    if min(width, height) < settings.min_side:
        raise ApiError(422, "image_too_small",
                       f"The image is too small; each side must be at least {settings.min_side} px.")
    try:
        image.load()  # decodes the pixel data; this is where truncated/corrupted files fail
    except Exception:  # noqa: BLE001  untrusted input: Pillow decoders raise many different exception types
        raise ApiError(422, "image_corrupted",
                       "The image could not be decoded; it may be corrupted or truncated.") from None
    return image.convert("RGB")


def preprocess(image: Image.Image, size: int) -> np.ndarray:
    """RGB image -> float32 array (1, 3, size, size) in [0, 1]."""
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.uint8).astype(np.float32) / np.float32(255.0)
    return np.ascontiguousarray(array.transpose(2, 0, 1))[None, ...]
