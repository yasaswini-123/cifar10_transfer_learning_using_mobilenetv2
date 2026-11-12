import os
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.model import build_model


IMAGE_SIZE = (96, 96)

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


# Build model and load weights once
model = build_model()
model.load_weights(WEIGHTS_PATH)


def predict_image(image_path):

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize image
    image = image.resize(IMAGE_SIZE)

    # Convert image to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # MobileNetV2 preprocessing
    image_array = preprocess_input(image_array)

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Predict
    predictions = model.predict(
        image_array,
        verbose=0
    )

    # Get predicted class
    predicted_index = np.argmax(predictions[0])

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        predictions[0][predicted_index]
    ) * 100

    return predicted_class, confidence


if __name__ == "__main__":

    image_path = input("Enter image path: ").strip()

    if not os.path.exists(image_path):
        print("Error: Image file not found!")
    else:
        label, confidence = predict_image(image_path)

        print("\n========== PREDICTION RESULT ==========")
        print("Predicted Class:", label)
        print("Confidence:", round(confidence, 2), "%")