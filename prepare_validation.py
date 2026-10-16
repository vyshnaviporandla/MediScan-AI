import random
import shutil
from pathlib import Path

ROOT = Path(r"D:\MedicalAI\data\raw\chest_xray")
CLASSES = ["NORMAL", "PNEUMONIA"]
EXTENSIONS = {".jpeg", ".jpg", ".png"}
SEED = 42

random.seed(SEED)

# Verify the dataset folders before making changes.
for split in ["train", "val", "test"]:
    for category in CLASSES:
        folder = ROOT / split / category
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder}")

# Select 10% from each training class for validation.
for category in CLASSES:
    train_dir = ROOT / "train" / category
    val_dir = ROOT / "val" / category

    images = [
        p for p in train_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    random.shuffle(images)
    count = max(1, round(len(images) * 0.10))
    selected = images[:count]

    for source in selected:
        destination = val_dir / source.name

        if destination.exists():
            raise FileExistsError(
                f"Destination already exists: {destination}"
            )

        shutil.move(str(source), str(destination))

    print(f"{category}: moved {count} images to validation")

print("\nValidation split prepared.")
print("Test images were not changed.")