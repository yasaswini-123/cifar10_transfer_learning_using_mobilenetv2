import os
import tensorflow as tf
import matplotlib.pyplot as plt

from src.data_loader import load_cifar10
from src.preprocessing import preprocess_dataset
from src.model import build_model


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 5
LEARNING_RATE = 0.00001
BATCH_SIZE = 16


def train_model():

    print("Loading CIFAR-10 dataset...")

    X_train, y_train, X_test, y_test = load_cifar10()

    print("Preparing training and validation datasets...")

    # Split training data into training and validation sets
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train
    )

    train_dataset, val_dataset = preprocess_dataset(
        X_train, y_train, X_val, y_val
    )

    print("Building MobileNetV2 model...")

    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    best_weights_path = os.path.join(
        MODEL_DIR, "best_model.weights.h5"
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=1,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print("\nStarting MobileNetV2 training...")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Load the best weights before final evaluation
    model.load_weights(best_weights_path)

    print("\nTraining completed!")

    # Accuracy graph
    plt.figure()

    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy.png"))
    plt.close()

    # Loss graph
    plt.figure()

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.savefig(os.path.join(OUTPUT_DIR, "loss.png"))
    plt.close()

    print("Best model weights saved successfully!")
    print("Training graphs saved successfully!")


if __name__ == "__main__":
    train_model()