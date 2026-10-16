
import os
import shutil
import zipfile
from pathlib import Path

archive = Path.home() / "Downloads" / "archive.zip"
destination = Path(r"D:\MedicalAI\data\raw\chest_xray")
prefix = "chest_xray/chest_xray/"
extensions = (".jpeg", ".jpg", ".png")

if not archive.exists():
    raise FileNotFoundError(f"Dataset ZIP not found: {archive}")

with zipfile.ZipFile(archive, "r") as z:
    images = [
        item for item in z.infolist()
        if not item.is_dir()
        and item.filename.startswith(prefix)
        and "__MACOSX" not in item.filename
        and item.filename.lower().endswith(extensions)
    ]

    print(f"Images to extract: {len(images)}")
    print(f"Destination: {destination}")
    print("Extraction started. Please wait...")

    for index, item in enumerate(images, start=1):
        relative_path = item.filename[len(prefix):]
        target = destination / Path(relative_path)

        target.parent.mkdir(parents=True, exist_ok=True)

        with z.open(item) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)

        if index % 500 == 0:
            print(f"Extracted {index}/{len(images)} images")

print("Dataset extraction completed.")
print(f"Dataset location: {destination}")
