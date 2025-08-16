# Disease Prediction Using Machine Learning

A machine learning project that predicts diseases based on patient symptoms. This project leverages Python, scikit-learn, and imbalanced-learn to train models such as Decision Tree, Random Forest, and SVM for accurate disease prediction.

## Overview

Predicting diseases early can help in timely treatment and reduce health risks. This project uses symptom data to train machine learning models, allowing users to input symptoms and get a predicted disease.

## Features

- Handles multiple diseases.
- Uses Stratified K-Fold cross-validation to evaluate models.
- Balances datasets using RandomOverSampler for better accuracy.
- Visualizes disease distribution using plots.
- Allows real-time predictions from sample input data.

## Dataset

The dataset `improved_disease_dataset.csv` contains the following fields:

- **Symptoms:** fever, cough, fatigue, headache, shortness_of_breath, etc.
- **Target:** disease (e.g., Flu, Covid, Cold, Malaria)

Example:

| fever | cough | fatigue | headache | shortness_of_breath | disease |
|-------|-------|---------|---------|-------------------|---------|
| 1     | 1     | 0       | 0       | 0                 | Flu     |
| 1     | 1     | 1       | 1       | 1                 | Covid   |
| 0     | 1     | 0       | 1       | 0                 | Cold    |

The dataset is placed inside the `data/` folder.

## Installation

Ensure you have Python 3.8+ installed. Install required dependencies:

```bash
pip install -r requirements.txt
## requirements:
numpy
pandas
scikit-learn
imbalanced-learn
matplotlib
seaborn
