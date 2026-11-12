import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from src.data_loader import load_cifar10
from src.preprocessing import preprocess_dataset
from src.model import build_model


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

WEIGHTS_PATH = os.path.join(
    BASE_DIR, "models", "best_model.weights.h5"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_model():

    # 1. Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")

    X_train, y_train, X_test, y_test = load_cifar10()

    print("Training images:", X_train.shape)
    print("Testing images:", X_test.shape)

    # 2. Prepare test dataset
    print("\nPreprocessing test dataset...")

    _, test_dataset = preprocess_dataset(
        X_train, y_train, X_test, y_test
    )

    # 3. Build MobileNetV2 architecture
    print("\nBuilding MobileNetV2 model...")

    model = build_model()

    # 4. Load Colab-trained weights
    print("Loading trained weights...")

    model.load_weights(WEIGHTS_PATH)

    print("Trained weights loaded successfully!")

    # 5. Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.00001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 6. Evaluate model
    print("\nEvaluating model...")

    loss, accuracy = model.evaluate(
        test_dataset,
        verbose=1
    )

    print("\n========== MODEL PERFORMANCE ==========")
    print("Test Loss:", round(loss, 4))
    print("Test Accuracy:", round(accuracy * 100, 2), "%")

    # 7. Generate predictions
    print("\nGenerating predictions...")

    predictions = model.predict(
        test_dataset,
        verbose=1
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    # 8. Classification report
    print("\n========== CLASSIFICATION REPORT ==========")

    report = classification_report(
        y_test,
        predicted_labels,
        target_names=CLASS_NAMES,
        digits=4
    )

    print(report)

    # Save classification report
    report_path = os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    )

    with open(report_path, "w") as file:
        file.write(report)

    # 9. Confusion matrix
    print("\nGenerating confusion matrix...")

    cm = confusion_matrix(
        y_test,
        predicted_labels
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cmap="Blues"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("CIFAR-10 Confusion Matrix")

    plt.tight_layout()

    confusion_path = os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nConfusion matrix saved:", confusion_path)
    print("Classification report saved:", report_path)

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    evaluate_model()