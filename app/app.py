import os
import numpy as np
import gradio as gr

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.model import build_model


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

WEIGHTS_PATH = os.path.join(
    BASE_DIR, "models", "best_model.weights.h5"
)

IMAGE_SIZE = (96, 96)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# Build model and load Colab-trained weights
model = build_model()
model.load_weights(WEIGHTS_PATH)

print("MobileNetV2 model loaded successfully!")


def classify_image(image):

    if image is None:
        return {}, "Please upload an image."

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = preprocess_input(image_array)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    results = {
        CLASS_NAMES[i]: float(predictions[i])
        for i in range(10)
    }

    predicted_index = np.argmax(predictions)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        predictions[predicted_index]
    ) * 100

    result_text = (
        f"Predicted Class: {predicted_class}\n"
        f"Confidence: {confidence:.2f}%"
    )

    return results, result_text


demo = gr.Interface(
    fn=classify_image,

    inputs=gr.Image(
        type="pil",
        label="Upload an Image"
    ),

    outputs=[
        gr.Label(
            num_top_classes=3,
            label="Prediction Results"
        ),

        gr.Textbox(
            label="Final Prediction"
        )
    ],

    title="CIFAR-10 Image Classification",

    description=(
        "Upload an image to classify it into "
        "one of the 10 CIFAR-10 categories "
        "using MobileNetV2."
    )
)


if __name__ == "__main__":
    demo.launch()