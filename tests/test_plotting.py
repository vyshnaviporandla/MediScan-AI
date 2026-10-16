"""Tests for presentation-ready plots."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.plotting import save_confusion_matrix, save_training_curves_from_csv


class PlottingTests(unittest.TestCase):
    def test_rebuilds_training_curves_from_existing_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            log_path = root / "training_log.csv"
            output_path = root / "training_curves.png"
            with log_path.open("w", newline="", encoding="utf-8") as log_file:
                writer = csv.DictWriter(
                    log_file,
                    fieldnames=["epoch", "accuracy", "loss", "val_accuracy", "val_loss"],
                )
                writer.writeheader()
                writer.writerow({
                    "epoch": 0,
                    "accuracy": 0.8,
                    "loss": 0.5,
                    "val_accuracy": 0.7,
                    "val_loss": 0.6,
                })

            save_training_curves_from_csv(log_path, output_path)

            self.assertGreater(output_path.stat().st_size, 0)

    def test_saves_confusion_matrix_with_matching_class_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "confusion_matrix.png"

            save_confusion_matrix(
                np.array([[92, 142], [6, 384]]),
                ["NORMAL", "PNEUMONIA"],
                output_path,
            )

            self.assertGreater(output_path.stat().st_size, 0)

    def test_rejects_mismatched_confusion_matrix_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "dimensions must match"):
            save_confusion_matrix(np.eye(2), ["NORMAL"], "unused.png")


if __name__ == "__main__":
    unittest.main()
