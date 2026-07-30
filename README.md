# Brain Tumor Detection Using Deep Learning

This project shows how to build a deep learning model that can classify brain MRI images into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

The notebook uses transfer learning with pre-trained image models and combines multiple models to improve prediction accuracy.

---

## Project Goal

The main purpose of this project is to create a reliable image classification system for medical MRI scans. The model learns patterns from brain images and predicts the correct tumor category for each scan.

---

## What the Notebook Does

The notebook follows a complete workflow:

1. Loads MRI image data from a dataset.
2. Organizes the images into train, validation, and test folders.
3. Prepares image data with augmentation techniques.
4. Builds a deep learning model using transfer learning.
5. Trains the model in stages for better performance.
6. Creates additional models and combines them into an ensemble.
7. Evaluates the final system using accuracy, precision, recall, F1-score, and a confusion matrix.

---

## Dataset

The notebook expects a brain MRI dataset with folders such as:

- Training/glioma
- Training/meningioma
- Training/notumor
- Training/pituitary
- Testing/glioma
- Testing/meningioma
- Testing/notumor
- Testing/pituitary

The original code uses a Kaggle dataset path:

- /kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset

If you are running the project locally, update the dataset path in the notebook to match your machine.

---

## Model Architecture

This project uses several powerful pre-trained image models:

### 1. EfficientNetB0
This is the main model. It is first trained with the new classification layers only, and then fine-tuned by unfreezing part of the base model.

### 2. ResNet50
A second model is built using ResNet50 to learn different visual features.

### 3. DenseNet121
A third model is built using DenseNet121 for more diversity in predictions.

### 4. Weighted Ensemble
The final prediction is made by combining the outputs of these three models. The EfficientNet model receives the highest weight because it performs best.

---

## Step-by-Step Workflow

### 1. Import Libraries
The notebook imports TensorFlow, Keras, NumPy, Matplotlib, Seaborn, and Scikit-learn.

### 2. Prepare the Dataset
Images are collected from the training and testing folders and split into:

- Training set
- Validation set
- Test set

### 3. Create Data Generators
The notebook uses image augmentation to improve generalization.

Examples include:

- Rotation
- Width and height shifting
- Zooming
- Horizontal flipping

### 4. Build the Model
A neural network is created using:

- EfficientNetB0 as the base model
- Global average pooling
- Dense layers
- Dropout layers
- A softmax output layer for 4 classes

### 5. Train the Model
Training happens in two phases:

#### Phase 1: Train the classification head
Only the newly added layers are trained first.

#### Phase 2: Fine-tune the base model
Part of the pre-trained model is unfrozen and trained again with a lower learning rate.

### 6. Train Additional Models
The notebook also trains ResNet50 and DenseNet121 to support the ensemble.

### 7. Evaluate Results
The final model is tested on the unseen test set and reports:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

---

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

Main libraries used:

- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow

---

## How to Run

1. Open the notebook file named brain-tumor-detection.ipynb.
2. Make sure the dataset path is correct.
3. Run all cells from top to bottom.
4. Wait for training to complete.
5. Review the results from the evaluation section.

If you are using Kaggle or Google Colab, the dataset path may already work without major changes.

---

## Output Files

The notebook produces and uses files such as:

- best_model.h5: the best trained model
- Confusion matrix visualization
- Training and validation plots
- Evaluation metrics in the notebook output

---

## Notes

- Training deep learning models is faster with a GPU, but CPU training is still possible.
- Transfer learning is used because it is faster and often more accurate than training a model from scratch.

---

## Summary

This notebook provides a complete deep learning pipeline for brain tumor MRI classification, including:

- Data preparation
- Image augmentation
- Transfer learning
- Model training
- Ensemble learning
- Performance evaluation

It is a strong example of how modern computer vision models can be applied to medical image analysis.
