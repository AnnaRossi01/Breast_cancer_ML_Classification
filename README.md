# Machine Learning pipeline for malignant breast cancer cells prediction

![Difference between Benign (left) and Malignant (right) cells](breastcancer_cells.png)

## Table of contents


# Project Overview and Workflow
--
This project applies various supervised machine learning algorithms to classify breast cancer tumors as benign or malignant using clinical data. It includes:

1. **Download and dataset preparation**
2. **Exploratory Data Analysis (EDA)**
   - Class Distribution
   - Density distribution of features
   - Boxplot of features distribution by diagnosis
   - Heatmap of Pearson Correlation matrix
3. **Data preparation**
   - Data rescaling (StandardScaler)
   - Feature selection (Univariate Selection & RFE)
4. **Training and evaluation of multiple models**:
    - Logistic Regression
    - LDA
    - KNN
    - Decision Tree
    - Naive Bayes
    - SVM
5. **Performance metrics**
     - Accuracy
     - Precision
     - Recall
     - Confusion Matrix
     - ROC-AUC
6. **Comparison of model performance**

The goal is to identify the best-performing algorithm for early and accurate detection of malignant tumors, contributing to better decision support in medical diagnostics.

## Dataset 
---
The dataset used is the **Breast Cancer Wisconsin (Diagnostic)** dataset, available from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)).
*Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.*

Each row represents a tumor instance with 30 numeric features and a binary diagnosis (`M = malignant`, `B = benign`).

**Features**: 
- **Perimeter**  
- **Area**
- **Radius** 
- **Smoothness** 
- **Compactness**
- **Concavity**
- **Concave Points**

## Tools used 
Data Handling --> pandas, numpy
Visualization --> matplotlib, seaborn
Modelling --> scikit-learn
Evaluation and matrices --> scikit-learn




 

