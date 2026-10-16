"""Tests for image loading and the stable binary label mapping."""
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import tensorflow as tf
from PIL import Image

from src.data import IMAGE_SIZE, load_split


class DataPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temporary_directory.name)
        for class_name, color in (("NORMAL", 32), ("PNEUMONIA", 224)):
            class_dir = self.data_root / "train" / class_name
            class_dir.mkdir(parents=True)
            Image.new("RGB", (16, 12), color=(color, color, color)).save(
                class_dir / "sample.png"
            )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_load_split_resizes_rgb_and_keeps_class_indices_stable(self) -> None:
        dataset = load_split(self.data_root, "train")
        images, labels = next(iter(dataset))

        self.assertEqual(tuple(images.shape[1:]), (*IMAGE_SIZE, 3))
        np.testing.assert_array_equal(labels.numpy().reshape(-1), [0.0, 1.0])
        self.assertTrue(tf.is_tensor(images))

    def test_missing_split_raises_actionable_error(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Missing"):
            load_split(self.data_root, "test")


if __name__ == "__main__":
    unittest.main()
