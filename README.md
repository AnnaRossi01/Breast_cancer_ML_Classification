# Machine Learning Pipeline for Malignant Breast Cancer Cell Prediction

![Difference between Benign (left) and Malignant (right) cells](brastcancer_cells.png)

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Tools Used](#tools-used)

---

## Project Overview

This project applies a complete supervised machine learning pipeline to classify breast cancer tumors as **benign** or **malignant**, using clinical diagnostic features.  
The objective is to identify the best-performing model to support **early and accurate tumor detection**, improving decision-making in medical diagnostics.

---

## Dataset

The dataset used is the **Breast Cancer Wisconsin (Diagnostic)** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)).

> *Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.*

Each row represents a tumor instance, with 30 numerical features derived from digitized images of a fine needle aspirate (FNA) of a breast mass.

- **Target variable**:  
  - `M` = Malignant  
  - `B` = Benign

- **Key features** include:  
  - Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension

---

## Workflow

1. **Data Acquisition & Preparation**
   - Dataset download
   - Cleaning and formatting

2. **Exploratory Data Analysis (EDA)**
   - Class distribution
   - Feature density plots
   - Boxplots by diagnosis
   - Correlation heatmap

3. **Feature Engineering**
   - Standardization (`StandardScaler`)
   - Feature selection (`SelectKBest`, `RFE`)

4. **Model Training & Evaluation**
   - Algorithms:
     - Logistic Regression
     - Linear Discriminant Analysis (LDA)
     - K-Nearest Neighbors (KNN)
     - Decision Tree
     - Naive Bayes
     - Support Vector Machine (SVM)
   - Evaluation metrics:
     - Accuracy
     - F1 Score
     - Precision & Recall
     - Confusion Matrix
     - ROC Curve & AUC

5. **Model Comparison**
   - Visual and tabular comparison of performance across models

---

## Tools Used

| Category         | Libraries                   |
|------------------|-----------------------------|
| **Data Handling**   | `pandas`, `numpy`             |
| **Visualization**   | `matplotlib`, `seaborn`       |
| **Modeling & Evaluation** | `scikit-learn`               |

---


## Author 
Rossi Anna

Project for AML Baic Course of Bioinformatics Master's degree at Bologna University

anna.rossi18@studio.unibo.it








 

