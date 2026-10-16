"""Tests for dataset inspection and preview generation."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.inspect_data import inspect_dataset, save_training_preview


class InspectDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temporary_directory.name)
        for split in ("train", "val", "test"):
            for class_name in ("NORMAL", "PNEUMONIA"):
                class_dir = self.data_root / split / class_name
                class_dir.mkdir(parents=True)
                Image.new("L", (12, 10), color=128).save(class_dir / "sample.png")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_counts_valid_images_and_reports_corrupt_images(self) -> None:
        corrupt_path = self.data_root / "train" / "NORMAL" / "broken.jpg"
        corrupt_path.write_bytes(b"not an image")

        inventory = inspect_dataset(self.data_root)

        self.assertEqual(len(inventory["train"]["NORMAL"].valid_files), 1)
        self.assertEqual(inventory["train"]["NORMAL"].corrupt_files, [corrupt_path])
        self.assertEqual(len(inventory["test"]["PNEUMONIA"].valid_files), 1)

    def test_saves_grayscale_preview(self) -> None:
        inventory = inspect_dataset(self.data_root)
        output_path = self.data_root / "outputs" / "preview.png"

        save_training_preview(inventory, output_path, samples_per_class=1)

        self.assertTrue(output_path.is_file())
        with Image.open(output_path) as preview:
            self.assertGreater(preview.width, 0)
            self.assertGreater(preview.height, 0)

    def test_missing_split_raises_actionable_error(self) -> None:
        (self.data_root / "test").rename(self.data_root / "missing-test")

        with self.assertRaisesRegex(FileNotFoundError, "Missing dataset split"):
            inspect_dataset(self.data_root)


if __name__ == "__main__":
    unittest.main()
