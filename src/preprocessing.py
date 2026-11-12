import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Configuration
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 16


def prepare_dataset(images, labels, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (images, labels)
    )

    # Shuffle training data
    if training:
        dataset = dataset.shuffle(buffer_size=5000)

    def preprocess_image(image, label):

        image = tf.cast(image, tf.float32)

        # Resize image
        image = tf.image.resize(
            image, IMAGE_SIZE
        )

        # MobileNetV2 preprocessing
        image = preprocess_input(image)

        return image, label

    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=1
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(1)

    return dataset


def preprocess_dataset(
    X_train, y_train, X_test, y_test
):

    train_dataset = prepare_dataset(
        X_train, y_train, training=True
    )

    test_dataset = prepare_dataset(
        X_test, y_test, training=False
    )

    return train_dataset, test_dataset


# TESTING SECTION
if __name__ == "__main__":

    from src.data_loader import load_cifar10

    X_train, y_train, X_test, y_test = load_cifar10()

    train_dataset, test_dataset = preprocess_dataset(
        X_train, y_train, X_test, y_test
    )

    print("\nPREPROCESSING TEST")
    print("--------------------------")

    for images, labels in train_dataset.take(1):

        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)

        print("Image data type:", images.dtype)

        print(
            "Minimum pixel value:",
            tf.reduce_min(images).numpy()
        )

        print(
            "Maximum pixel value:",
            tf.reduce_max(images).numpy()
        )

    print("Preprocessing working successfully!")