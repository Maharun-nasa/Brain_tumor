# Brain Tumor Detection with Deep Learning

This project uses deep learning to classify brain MRI images into four classes: glioma, meningioma, no tumor, and pituitary tumor.

## Overview

The notebook builds a medical image classification pipeline using transfer learning and ensemble modeling. It is designed to detect tumor patterns from MRI scans with strong accuracy and clear evaluation results.

## Simple Dataset Structure

Place the data in a simple folder layout like this:

```text
brain_tumor_dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

The notebook originally uses a Kaggle-style path:

```text
/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset
```

If you are running it locally, update the dataset path in the notebook to match your machine.

## Models Used

The project uses:

- EfficientNetB0 as the main model
- ResNet50 as a second model
- DenseNet121 as a third model
- A weighted ensemble to combine predictions

## Results from the Notebook

The notebook outputs show strong performance:

- EfficientNetB0 test accuracy: 97.69%
- EfficientNetB0 precision: 0.9769
- EfficientNetB0 recall: 0.9769
- EfficientNetB0 F1-score: 0.9768
- Training accuracy: 99.68%
- Validation accuracy: 98.61%
- Test accuracy: 97.69%
- Ensemble test accuracy: 97.31%
- ResNet50 test accuracy: 93.24%
- DenseNet121 test accuracy: 84.54%

These results show that the model performs very well on unseen MRI data.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## How to Run

1. Open the notebook file: brain-tumor-detection.ipynb
2. Make sure the dataset path is correct
3. Run all cells from top to bottom
4. Review the outputs, plots, and evaluation metrics

## Output Files

The notebook produces:

- best_model.h5
- confusion matrix plots
- training and validation accuracy plots
- evaluation metrics in the notebook output

## Notes

- GPU training is recommended for faster performance
- Transfer learning is used to get strong results with less training time
