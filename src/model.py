"""Transfer-learning model. Scores are model outputs, not clinical probabilities."""
import tensorflow as tf


def build_model(input_shape=(224, 224, 3)):
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.04),
        tf.keras.layers.RandomZoom(0.08),
    ], name="training_augmentation")
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=input_shape, name="xray_rgb")
    x = augmentation(inputs)
    # MobileNetV2 expects RGB values mapped to [-1, 1].
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pneumonia_score")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    return model
