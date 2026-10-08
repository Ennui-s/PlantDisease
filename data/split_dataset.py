import os
import shutil
import random
from config import CONFIG

def split_dataset():
    random.seed(42)
    raw_dir = CONFIG["raw_dir"]
    output_dir = CONFIG["data_dir"]
    train_ratio = CONFIG["train_ratio"]
    val_ratio = CONFIG["val_ratio"]
    classes = os.listdir(raw_dir)
    for cls in classes:
        class_path = os.path.join(raw_dir, cls)
        images = os.listdir(class_path)
        random.shuffle(images)
        train_cut = int(len(images) * train_ratio)
        val_cut = int(len(images) * (train_ratio + val_ratio))
        splits = {
            "train": images[:train_cut],
            "val": images[train_cut:val_cut],
            "test": images[val_cut:]
        }
        for split in splits:
            split_path = os.path.join(output_dir, split, cls)
            os.makedirs(split_path, exist_ok=True)

            for img in splits[split]:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_path, img)
                shutil.copy(src, dst)

if __name__ == "__main__":
    split_dataset()