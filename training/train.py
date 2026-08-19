from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "gtsrb" / "processed"
MODEL_DIR = PROJECT_DIR / "modelo"

IMAGE_SIZE = (32, 32)
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
SEED = 42


# ============================================================
# DATASET
# ============================================================

def load_datasets():
    train = keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=SEED,
    )

    val = keras.utils.image_dataset_from_directory(
        DATA_DIR / "val",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    test = keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    augmentation = keras.Sequential([
        layers.RandomRotation(10 / 360),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomZoom(0.10),
        layers.RandomBrightness(0.10, value_range=(0, 1)),
        layers.RandomContrast(0.10),
    ])

    def train_preprocess(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        images = augmentation(images, training=True)
        images = tf.clip_by_value(images, 0.0, 1.0)
        return images, labels

    def test_preprocess(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        return images, labels

    train = train.map(train_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val = val.map(test_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test = test.map(test_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    train = train.prefetch(tf.data.AUTOTUNE)
    val = val.prefetch(tf.data.AUTOTUNE)
    test = test.prefetch(tf.data.AUTOTUNE)

    return train, val, test


# ============================================================
# MODEL
# ============================================================

def create_model():
    inputs = keras.Input(shape=(32, 32, 3))

    x = layers.Conv2D(16, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)

    outputs = layers.Dense(43)(x)

    return keras.Model(inputs, outputs, name="TrafficSignNet")


# ============================================================
# TRAIN
# ============================================================

def main():
    keras.utils.set_random_seed(SEED)

    train_ds, val_ds, test_ds = load_datasets()

    model = create_model()
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    loss, accuracy = model.evaluate(test_ds)

    print("\n============================")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print("============================")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "TrafficSignNet_FP32.h5")

    print(f"Model saved in: {MODEL_DIR / 'TrafficSignNet_FP32.h5'}")


if __name__ == "__main__":
    main()
