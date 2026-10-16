"""Predict one image. This demo output is not a medical diagnosis."""
import argparse
from pathlib import Path
import tensorflow as tf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path)
    p.add_argument("--model", type=Path, default=Path("models/pneumonia_mobilenet.keras"))
    args = p.parse_args()
    image = tf.keras.utils.load_img(args.image, target_size=(224, 224), color_mode="rgb")
    tensor = tf.keras.utils.img_to_array(image)[None, ...]
    score = float(tf.keras.models.load_model(args.model).predict(tensor, verbose=0)[0][0])
    label = "PNEUMONIA pattern" if score >= 0.5 else "NORMAL pattern"
    print(f"Demo label: {label}; model score: {score:.3f}")
    print("Educational research demonstration only. Not a diagnosis or clinical decision.")


if __name__ == "__main__":
    main()
