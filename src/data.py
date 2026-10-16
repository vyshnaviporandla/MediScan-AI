"""Input pipeline for a directory-organized binary image dataset."""
from pathlib import Path
import tensorflow as tf

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def load_split(root: Path | str, split: str, training: bool = False):
    """Load one split and enforce a stable class-index mapping."""
    root = Path(root)
    directory = root / split
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing {directory}. See data/README.md for setup.")
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=training,
        seed=42,
    )
    return ds.prefetch(tf.data.AUTOTUNE)
