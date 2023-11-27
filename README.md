# Real-Time MRI Brain Tumor Classification Project

## Overview

This project focuses on the development of a deep learning model for classifying MRI brain scans into four categories: normal, glioma, meningioma, and pituitary tumors. The primary objective is to create a reliable classifier that ensures precise identification of these brain tumor types while minimizing false positives in medical diagnostics.

## Table of Contents

1. [Dataset](#dataset)
2. [Models](#models)
3. [Notebooks](#notebooks)
4. [Scripts](#scripts)
5. [Templates](#templates)
6. [Requirements](#requirements)
7. [Model Deployment](#model-deployment)
8. [Summary of Analysis](#summary-of-analysis)
9. [References](#references)

## Dataset

The dataset used for training and evaluation is stored in the `/datasets` directory. It includes the following zip files:

- `/datasets/train_dataset.zip`: Training dataset
- `/datasets/val_dataset.zip`: Validation dataset
- `/datasets/test_dataset.zip`: Test dataset

Please unzip these files before using them for training and evaluation.

## Models

The trained model file and its corresponding weights are stored in the `/models` directory:

- `/models/resnet50.pth`: PyTorch weights for the trained ResNet50 model (referred to as Model 1 in the project)


## Notebooks

The Jupyter notebook containing the code for training and evaluating the models is stored in the `/notebooks` directory:

- `/notebooks/real_time_brain_tumor_classification.ipynb`

## Scripts

The scripts for model deployment and the custom model class are stored in the `/scripts` directory:

- `/scripts/main.py`: FastAPI script for model deployment
- `/scripts/custom_model.py`: Script containing the ResNet50 custom model class

## Templates

The HTML template for the FastAPI web interface is stored in the `/templates` directory:

- `/templates/home.html`

## Requirements

The required Python packages and dependencies are listed in the `requirements.txt` file.

## Model Deployment

### How to Access the Model

To access the deployed model, follow these steps:

1. Run the FastAPI script using `python main.py`.
2. Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.
3. Upload an image for brain tumor classification.

The model will provide classification results, including class label, class name, and confidence.

Note: The FastAPI web app must be running locally to access the model via the provided link. Others will require the code and a local environment to run the app themselves.

## Summary of Analysis

The analysis assessed various models using key metrics, including accuracy, specificity, sensitivity, precision, and F1-score. Findings and model comparisons are presented in the Jupyter notebook.

## References

Dubail, T. (2023). Brain Tumors (256x256) [Dataset]. Kaggle. [Link to Dataset](https://www.kaggle.com/datasets/thomasdubail/brain-tumors-256x256)
