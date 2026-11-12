import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    RandomFlip,
    RandomRotation,
    RandomTranslation
)
from tensorflow.keras.models import Model


def build_model():

    inputs = Input(shape=(96, 96, 3))

    x = RandomFlip("horizontal")(inputs)
    x = RandomRotation(0.04)(x)
    x = RandomTranslation(
        height_factor=0.1,
        width_factor=0.1
    )(x)

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(96, 96, 3),
        name="mobilenetv2_base"
    )

    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):
            layer.trainable = False

    x = base_model(x, training=False)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(10, activation="softmax")(x)

    return Model(inputs, outputs)