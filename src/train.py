"""Train and save the educational baseline model."""
import argparse
from pathlib import Path
import tensorflow as tf
from src.data import load_split
from src.model import build_model
from src.plotting import save_training_curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/chest_xray"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("models/pneumonia_mobilenet.keras"))
    args = parser.parse_args()
    data_root = Path(args.data)
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    train_ds = load_split(data_root, "train", training=True)
    val_ds = load_split(data_root, "val")
    model = build_model()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.output, monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(output_dir / "training_log.csv"),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    save_training_curves(history.history, output_dir / "training_curves.png")
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
