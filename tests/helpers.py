"""Small builders shared by the tests."""
import io

import numpy as np
from PIL import Image

_rng = np.random.default_rng(0)


def image_bytes(fmt="JPEG", size=(256, 256), mode="RGB"):
    """A random-noise image encoded as bytes (every call gives a different image)."""
    arr = _rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).convert(mode).save(buf, format=fmt)
    return buf.getvalue()


def make_dataset(root, classes=None, per_class=12, ext=".jpg"):
    """<root>/<class>/<n>.jpg with distinct random images."""
    classes = classes or {"class_a": per_class, "class_b": per_class}
    for name, n in classes.items():
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (folder / f"{i}{ext}").write_bytes(image_bytes())
    return root
