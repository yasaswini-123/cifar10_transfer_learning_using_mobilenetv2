import os
from src.model import build_model

MODEL_PATH = os.path.join(
    "models",
    "best_model.weights.h5"
)

# Build the MobileNetV2 architecture
model = build_model()

# Load trained weights from Colab
model.load_weights(MODEL_PATH)

print("Model architecture loaded successfully!")
print("Colab-trained weights loaded successfully!")

# Test with one sample image
import numpy as np

sample = np.random.rand(1, 96, 96, 3).astype("float32")

prediction = model.predict(sample)

print("Prediction shape:", prediction.shape)
print("Model test successful!")