"""Service settings, read from environment variables so the same code runs locally and in a container."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Settings:
    tracking_uri: str = "sqlite:///mlflow.db"       # MLFLOW_TRACKING_URI
    model_name: str = "plant-disease-classifier"    # MODEL_NAME
    model_alias: str = "champion"                   # MODEL_ALIAS
    max_upload_bytes: int = 5_000_000               # MAX_UPLOAD_BYTES (the training images are ~20 KB)
    max_pixels: int = 25_000_000                    # MAX_PIXELS: reject huge images before decoding them
    min_side: int = 32                              # MIN_SIDE: same minimum as the training data validation
    allowed_formats: Tuple[str, ...] = ("JPEG", "PNG")
    default_input_size: int = 224                   # used only if the model does not declare its input size
    top_k: int = 3                                  # TOP_K
    admin_token: Optional[str] = None               # ADMIN_TOKEN: enables POST /reload when set

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        d = cls()
        return cls(
            tracking_uri=env.get("MLFLOW_TRACKING_URI", d.tracking_uri),
            model_name=env.get("MODEL_NAME", d.model_name),
            model_alias=env.get("MODEL_ALIAS", d.model_alias),
            max_upload_bytes=int(env.get("MAX_UPLOAD_BYTES", d.max_upload_bytes)),
            max_pixels=int(env.get("MAX_PIXELS", d.max_pixels)),
            min_side=int(env.get("MIN_SIDE", d.min_side)),
            top_k=int(env.get("TOP_K", d.top_k)),
            admin_token=env.get("ADMIN_TOKEN") or None,
        )
