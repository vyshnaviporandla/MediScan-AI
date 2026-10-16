"""Inspect dataset splits and create a grayscale training-image preview."""
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

SPLITS = ("train", "val", "test")
CLASS_NAMES = ("NORMAL", "PNEUMONIA")
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}


@dataclass
class ClassInventory:
    """Image files found for one class in one dataset split."""

    valid_files: list[Path]
    corrupt_files: list[Path]


def inspect_dataset(data_root: Path) -> dict[str, dict[str, ClassInventory]]:
    """Verify supported image files and count valid/corrupt files by split."""
    inventory: dict[str, dict[str, ClassInventory]] = {}
    for split in SPLITS:
        split_dir = data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Missing dataset split: {split_dir}. See data/README.md for setup."
            )

        inventory[split] = {}
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing class directory: {class_dir}. "
                    "Expected NORMAL and PNEUMONIA for every split."
                )

            valid_files: list[Path] = []
            corrupt_files: list[Path] = []
            for image_path in sorted(class_dir.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except OSError:
                    corrupt_files.append(image_path)
                else:
                    valid_files.append(image_path)

            inventory[split][class_name] = ClassInventory(
                valid_files=valid_files,
                corrupt_files=corrupt_files,
            )
    return inventory


def save_training_preview(
    inventory: dict[str, dict[str, ClassInventory]],
    output_path: Path,
    samples_per_class: int = 3,
) -> None:
    """Save a deterministic grayscale montage from valid training images."""
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be at least 1.")

    train_inventory = inventory["train"]
    if not any(
        train_inventory[class_name].valid_files for class_name in CLASS_NAMES
    ):
        raise ValueError("No valid training images are available for the preview.")

    figure, axes = plt.subplots(
        len(CLASS_NAMES),
        samples_per_class,
        figsize=(3 * samples_per_class, 6),
        squeeze=False,
    )
    for row, class_name in enumerate(CLASS_NAMES):
        files = train_inventory[class_name].valid_files[:samples_per_class]
        for column, axis in enumerate(axes[row]):
            axis.axis("off")
            if column >= len(files):
                if column == 0 and not files:
                    axis.text(0.5, 0.5, "No valid images", ha="center", va="center")
                continue
            with Image.open(files[column]) as image:
                axis.imshow(image.convert("L"), cmap="gray")
            axis.set_title(class_name)

    figure.suptitle("Training-set image preview (grayscale display)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check dataset folders, count images, and save a grayscale preview."
    )
    parser.add_argument("--data", type=Path, default=Path("data/raw/chest_xray"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dataset_preview.png"),
    )
    parser.add_argument("--samples-per-class", type=int, default=3)
    args = parser.parse_args()
    if args.samples_per_class < 1:
        parser.error("--samples-per-class must be at least 1")

    inventory = inspect_dataset(args.data)
    corrupt_count = 0
    print("Dataset image inventory (supported image files only)")
    for split in SPLITS:
        print(f"\n{split}:")
        for class_name in CLASS_NAMES:
            class_inventory = inventory[split][class_name]
            valid_count = len(class_inventory.valid_files)
            corrupt_files = class_inventory.corrupt_files
            corrupt_count += len(corrupt_files)
            print(
                f"  {class_name}: {valid_count} valid, "
                f"{len(corrupt_files)} unreadable"
            )
            for path in corrupt_files:
                print(f"    Unreadable image: {path}")

    save_training_preview(inventory, args.output, args.samples_per_class)
    print(f"\nSaved grayscale preview to {args.output}")
    if corrupt_count:
        print(f"Warning: found {corrupt_count} unreadable image(s); review before training.")


if __name__ == "__main__":
    main()
