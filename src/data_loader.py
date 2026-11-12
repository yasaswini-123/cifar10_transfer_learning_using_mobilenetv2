import os
import tarfile
import pickle
import numpy as np


# Project paths
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(BASE_DIR, "data")

TAR_PATH = os.path.join(
    DATA_DIR, "cifar-10-python.tar.gz"
)

EXTRACT_PATH = DATA_DIR


def extract_dataset():

    extracted_folder = os.path.join(
        DATA_DIR, "cifar-10-batches-py"
    )

    if not os.path.exists(extracted_folder):

        print("Extracting CIFAR-10 dataset...")

        with tarfile.open(TAR_PATH, "r:gz") as tar:
            tar.extractall(path=EXTRACT_PATH)

        print("Dataset extracted successfully!")

    else:
        print("Dataset already extracted!")


def load_batch(file_path):

    with open(file_path, "rb") as file:
        batch = pickle.load(file, encoding="bytes")

    images = batch[b"data"]
    labels = batch[b"labels"]

    # Reshape images into RGB format
    images = images.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)

    return images, np.array(labels)


def load_cifar10():

    extract_dataset()

    dataset_path = os.path.join(
        DATA_DIR, "cifar-10-batches-py"
    )

    X_train = []
    y_train = []

    # Load five training batches
    for i in range(1, 6):

        file_path = os.path.join(
            dataset_path, f"data_batch_{i}"
        )

        images, labels = load_batch(file_path)

        X_train.append(images)
        y_train.extend(labels)

    X_train = np.concatenate(X_train)
    y_train = np.array(y_train)

    # Load test batch
    test_path = os.path.join(
        dataset_path, "test_batch"
    )

    X_test, y_test = load_batch(test_path)

    return X_train, y_train, X_test, y_test


# TESTING SECTION
if __name__ == "__main__":

    X_train, y_train, X_test, y_test = load_cifar10()

    print("\nDATA LOADER TEST")
    print("--------------------------")

    print("Training images:", X_train.shape)
    print("Training labels:", y_train.shape)

    print("Testing images:", X_test.shape)
    print("Testing labels:", y_test.shape)

    print("Data type:", X_train.dtype)
    print("Number of classes:", len(np.unique(y_train)))

    print("Data loader working successfully!")