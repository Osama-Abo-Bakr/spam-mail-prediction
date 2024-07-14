# Spam Mail Prediction

## Project Overview

This project aims to predict whether an email is spam or not using machine learning techniques. The project involves data preprocessing, feature extraction, model training, and evaluation to achieve high accuracy in spam detection.

## Table of Contents

1. [Introduction](#introduction)
2. [Libraries and Tools](#libraries-and-tools)
3. [Data Analysis and Preprocessing](#data-analysis-and-preprocessing)
4. [Feature Extraction](#feature-extraction)
5. [Modeling](#modeling)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Results](#results)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Conclusion](#conclusion)
11. [Best Estimator](#Best_Estimator)


## Introduction

The goal of this project is to classify emails as spam or ham (not spam) using machine learning models. The dataset consists of email messages and their corresponding labels (spam or ham).

## Libraries and Tools

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib, Seaborn**: Data visualization
- **Scikit-learn**: Preprocessing, feature extraction, and model building
- **XGBoost**: Advanced machine learning model

## Data Analysis and Preprocessing

1. **Data Loading**:
   - Loaded the dataset using `pd.read_csv()`.

2. **Data Summary**:
   - Checked for missing values using `data.isnull().sum()`.
   - Summarized the data using `data.info()` and `data.shape()`.

3. **Label Encoding**:
   - Converted categorical labels (spam, ham) to numerical labels (1, 0) using `data.loc[]`.

4. **Data Splitting**:
   - Split the data into training and testing sets using `train_test_split()`.

## Feature Extraction

1. **TF-IDF Vectorization**:
   - Converted the text data into numerical features using `TfidfVectorizer()`.

## Modeling

1. **Logistic Regression**:
   - Built a Logistic Regression model using `LogisticRegression()`.

2. **XGBoost Classifier**:
   - Built an XGBoost model using `xgb.XGBClassifier()`.

## Hyperparameter Tuning

1. **GridSearchCV**:
   - Tuned hyperparameters for the XGBoost model using `GridSearchCV()` to find the best parameters.

## Results

- **Model Accuracy**:
  - Achieved high accuracy in both Logistic Regression and XGBoost models.
  - The best hyperparameters for XGBoost were identified using GridSearchCV.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/spam-mail-prediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd spam-mail-prediction
   ```

## Usage

1. **Prepare Data**:
   - Ensure the dataset is available at the specified path.

2. **Train Models**:
   - Run the provided script to train the models and evaluate their performance.

3. **Predict Outcomes**:
   - Use the trained models to predict whether new emails are spam or ham.

## Conclusion

This project demonstrates the use of machine learning techniques for spam email detection. By leveraging data preprocessing, feature extraction, and model tuning, high accuracy was achieved in classifying emails.

## Best_Estimator

```python
Fitting 10 folds for each of 250 candidates, totalling 2500 fits
GridSearchCV(cv=10,
             estimator=XGBClassifier(base_score=None, booster=None,
                                     callbacks=None, colsample_bylevel=None,
                                     colsample_bynode=None,
                                     colsample_bytree=None,
                                     early_stopping_rounds=None,
                                     enable_categorical=False, eval_metric=None,
                                     feature_types=None, gamma=None,
                                     gpu_id=None, grow_policy=None,
                                     importance_type=None,
                                     interaction_constraints=None,
                                     learning_rate=None...
                                     max_delta_step=None, max_depth=1000,
                                     max_leaves=10, min_child_weight=None,
                                     missing=nan, monotone_constraints=None,
                                     n_estimators=10, n_jobs=None,
                                     num_parallel_tree=None, predictor=None,
                                     random_state=None, ...),
             n_jobs=-1,
             param_grid={'max_depth': [10, 30, 50, 70, 90],
                         'max_leaves': [10, 30, 50, 70, 90],
                         'n_estimators': [10, 30, 50, 70, 90, 110, 130, 150,
                                          170, 190]},
             scoring='accuracy', verbose=6)
```

---

### Sample Code (for reference)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Reading Data
data = pd.read_csv(r"D:\Courses language programming\5_Machine Learning\Dataset For Machine Learning\Spam_Mail\mail_data.csv")
data.head(5)

# Data Preprocessing
data.isnull().sum()
data["Category"].value_counts()
data.loc[data["Category"] == "spam", "Category"] = 1
data.loc[data["Category"] == "ham", "Category"] = 0
data.head(5)

x_input = data["Message"]
y_output = data["Category"]
x_train, x_test, y_train, y_test = train_test_split(x_input, y_output, train_size=0.7, random_state=42)

# Feature Extraction
feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
new_x_train = feature_extraction.fit_transform(x_train)
new_x_test = feature_extraction.transform(x_test)
y_train = y_train.astype("int")
y_test = y_test.astype("int")

# Building Models
# Logistic Regression
model1 = LogisticRegression()
model1.fit(new_x_train, y_train)
print(f"The Accuracy Training Data Score is {model1.score(new_x_train, y_train)}")
print(f"The Accuracy Testing Data Score is {model1.score(new_x_test, y_test)}")

# XGBoost Classifier
model2 = xgb.XGBClassifier()
model2.fit(new_x_train, y_train)
print(f"The Accuracy Training Data Score is {model2.score(new_x_train, y_train)}")
print(f"The Accuracy Testing Data Score is {model2.score(new_x_test, y_test)}")

# Hyperparameter Tuning with GridSearchCV
param_grid = {"n_estimators": list(range(10, 200, 20)), "max_depth": list(range(10, 100, 20)), "max_leaves": list(range(10, 100, 20))}
model_grid = GridSearchCV(estimator=xgb.XGBClassifier(n_estimators=10, max_depth=1000, max_leaves=10), param_grid=param_grid, cv=10, verbose=6, n_jobs=-1, scoring="accuracy")
model_grid.fit(new_x_train, y_train)
print(f"The Best Param is {model_grid.best_params_}")
print("-" * 50)
print(f"The Accuracy Training Data is {model_grid.score(new_x_train, y_train)}")
print(f"The Accuracy Testing Data is {model_grid.score(new_x_test, y_test)}")
```
