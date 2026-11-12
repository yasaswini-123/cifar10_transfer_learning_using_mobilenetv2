# CIFAR-10 Image Classification Using MobileNetV2

## Project Overview

This project implements an image classification system using **Transfer Learning with MobileNetV2** to classify images into the 10 categories of the CIFAR-10 dataset.

The model uses ImageNet-pretrained MobileNetV2 as its feature extraction backbone and is fine-tuned for CIFAR-10 image classification.

A Gradio web application allows users to upload an image and receive its predicted class along with confidence scores.

The trained model achieved **87.34% test accuracy** with a test loss of **0.3907**.

**The trained model weights are provided, so users can run the application directly without training the model again.**

---

## Project Objectives

- Implement image classification using Transfer Learning.
- Use pretrained MobileNetV2 for feature extraction.
- Fine-tune the model for CIFAR-10 classification.
- Apply image preprocessing and data augmentation.
- Evaluate model performance using accuracy, loss, a confusion matrix, and a classification report.
- Develop an interactive image classification application using Gradio.

---

## Dataset

**CIFAR-10 Dataset**

The CIFAR-10 dataset contains 60,000 colour images of size 32 × 32 pixels, divided into 10 classes.

| Dataset Details | Information |
|---|---|
| Total images | 60,000 |
| Training images | 50,000 |
| Testing images | 10,000 |
| Image dimensions | 32 × 32 × 3 |
| Number of classes | 10 |

### CIFAR-10 Classes

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

### Dataset Download

Download the CIFAR-10 Python dataset from:

https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

Extract it into the project's `data/` directory.

---

## Model Architecture

### MobileNetV2 – Transfer Learning

The project uses MobileNetV2 pretrained on ImageNet.

The following steps are performed:

1. Load the pretrained MobileNetV2 model without its original classification head.
2. Resize CIFAR-10 images to 96 × 96 pixels.
3. Apply MobileNetV2 preprocessing.
4. Apply data augmentation:
   - Random horizontal flipping
   - Random rotation
   - Random translation
5. Fine-tune selected layers of MobileNetV2.
6. Add Global Average Pooling.
7. Add a Dense layer with 128 neurons and ReLU activation.
8. Apply Dropout with a rate of 0.3.
9. Add a final Dense layer with 10 output neurons and Softmax activation.

### Training Configuration

| Parameter | Value |
|---|---|
| Model | MobileNetV2 |
| Pretrained weights | ImageNet |
| Input image size | 96 × 96 × 3 |
| Batch size | 16 |
| Optimizer | Adam |
| Learning rate | 0.00001 |
| Loss function | Sparse Categorical Crossentropy |
| Dropout | 0.3 |
| Output classes | 10 |

---

## Model Performance

The trained MobileNetV2 model was evaluated on the CIFAR-10 test dataset.

### Test Results

| Evaluation Metric | Result |
|---|---:|
| Test Accuracy | **87.34%** |
| Test Loss | **0.3907** |

The model correctly classifies approximately 87 out of every 100 test images.

### Evaluation Outputs

The project generates:

- **Confusion Matrix:** Visualizes correct and incorrect predictions across the 10 classes.
- **Classification Report:** Provides precision, recall, F1-score, and support for each class.

Evaluation results are saved in the `outputs/` directory.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- Gradio

---

## Project Structure

## Project Structure

```text
cifar10_transfer_learning_using_mobilenetv2/
│
├── app/
│   └── app.py
│
├── data/                         # Dataset (download separately)
│
├── models/
│   └── best_model.weights.h5     # Trained MobileNetV2 weights
│
├── outputs/
│   ├── classification_report.txt
│   └── confusion_matrix.png
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── test_model.py
│
├── training_mobilenetv2_collab/
│   └── cifar10_mobilenetv2_training_ipynb.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```
---

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project directory:

```bash
cd cifar10_transfer_learning
```

### Step 2: Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate the environment on Ubuntu/Linux:

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download or Check the Trained Model Weights

Make sure the following file is available:

```text
models/best_model.weights.h5
```

If the weights are hosted separately, download them and place them inside the `models/` directory.

**Note:** Users do not need to train the model again to run the application.

### Step 5: Run the Gradio Application

From the project root directory, execute:

```bash
python -m app.app
```

Open the following address in your browser:

```text
http://127.0.0.1:7860
```

### Step 6: Make Predictions

1. Open the Gradio application.
2. Upload an image.
3. The application preprocesses the image.
4. MobileNetV2 predicts the image category.
5. The application displays the top three predictions and confidence scores.

---

## Model Training

The model was trained using **Google Colab**.

The training notebook is included in this repository:

`training_mobilenetv2_colab.ipynb`

Users who want to reproduce or modify the training process can:

1. Open the notebook in Google Colab.
2. Upload or download the CIFAR-10 dataset.
3. Execute the preprocessing steps.
4. Build and compile the MobileNetV2 model.
5. Train the model.
6. Save and download the trained model weights.
7. Place the downloaded weights in the local `models/` directory.

### Local Training (Optional)

A local training script is also provided:

```bash
python -m src.train
```

Training may require significant computing resources. Google Colab is the recommended environment for training.

**Important:** Training is optional. Users can directly launch the application using the provided trained weights.

---

## Model Evaluation

To evaluate the trained model, run:

```bash
python -m src.evaluate
```

The evaluation module calculates:

- Test loss
- Test accuracy
- Precision
- Recall
- F1-score

It also generates a confusion matrix and classification report.

The results are saved in the `outputs/` directory.

---

## Single Image Prediction

To classify an individual image using the command line:

```bash
python -m src.predict
```

Enter the image path when prompted.

The program displays the predicted class and its confidence score.

---

## Application Features

- Simple and interactive Gradio interface.
- Upload images for classification.
- Classify images into 10 CIFAR-10 categories.
- Display the top three predictions.
- Show prediction confidence scores.
- Use the saved MobileNetV2 weights without retraining.

---

## Hardware Requirements

- Python 3.10 or a compatible Python version.
- TensorFlow and Keras versions compatible with the model architecture and saved weights.
- Required packages listed in `requirements.txt`.
- Internet access may be needed initially to download ImageNet weights if they are not cached.

A GPU is recommended for training. CPU-based inference is supported, although it may be slower.

---

## Future Improvements

- Improve classification accuracy through further fine-tuning.
- Experiment with other pretrained architectures.
- Optimize inference speed.
- Deploy the application online.
- Extend the system to support additional image categories.

---

## Author

Developed as a Deep Learning and Transfer Learning project using MobileNetV2 and the CIFAR-10 dataset.
