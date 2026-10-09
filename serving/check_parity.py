"""Check that the service's preprocessing matches the training pipeline, pixel for pixel.

    python -m serving.check_parity [--image dataset/test/<class>/<file>.jpg] [--n 20]

Applies data/dataloader.py's transform (torchvision Resize + ToTensor on an RGB PIL image) and the
service's preprocess() to the same files and compares the arrays.
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

from config import CONFIG
from serving.images import load_image, preprocess
from serving.settings import Settings


def main(argv=None) -> int:
    from torchvision import transforms

    parser = argparse.ArgumentParser(description="Compare serving preprocessing with the training transform.")
    parser.add_argument("--image", default=None, help="one image; default: random images from the test split")
    parser.add_argument("--n", type=int, default=20, help="number of test images to compare")
    args = parser.parse_args(argv)

    size = CONFIG["image_size"]
    train_tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    if args.image:
        paths = [Path(args.image)]
    else:
        files = sorted(p for p in (Path(CONFIG["data_dir"]) / "test").rglob("*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        paths = random.Random(0).sample(files, min(args.n, len(files)))
    if not paths:
        print("No images found.")
        return 2

    settings, worst = Settings(), 0.0
    for p in paths:
        expected = train_tf(Image.open(p).convert("RGB")).numpy()[None, ...]
        actual = preprocess(load_image(p.read_bytes(), settings), size)
        if expected.shape != actual.shape:
            print(f"SHAPE MISMATCH for {p}: {expected.shape} vs {actual.shape}")
            return 1
        worst = max(worst, float(np.abs(expected - actual).max()))
    print(f"Compared {len(paths)} image(s): max absolute difference {worst:.2e}")
    print("IDENTICAL preprocessing" if worst == 0.0 else
          "Preprocessing differs from training" if worst > 1e-6 else "Equal to float rounding")
    return 0 if worst <= 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
