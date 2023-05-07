# Diabetes-Status-Classification

## Project Overview

This repository contains a machine learning project focused on classification. 

The objective of this project is to predict the likelihood of diabetes in patients based on their medical history and demographic details. To accomplish this, we conduct an in-depth analysis of various features such as `age`, `gender`, `hypertension`, `BMI`, `blood glucose levels`, and others, utilizing different classifiers such as **Logistic Regression, SVM, K-Nearest Neighbors, Random Forests, and XGBoost**. We assess their performance by tuning different hyperparameters and comparing their results. We evaluate the classifiers' performance using metrics such as **accuracy, precision, recall, F1 score, Confusion Matrix and ROC**.

## Dataset

The diabetes prediction dataset is stored in the "diabetes_prediction_dataset.csv" file, which includes medical and demographic data of patients and their corresponding diabetes status, either positive or negative. The dataset consists of several features, including age, gender, body mass index (BMI), hypertension, heart disease, smoking history, HbA1c level, and blood glucose level.
You may also download it from [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)


## Software And Tools Requirements

1. [Github Account](https://github.com)
2. [GitCLI](https://git-scm.com/downloads)
3. [VS Code IDE](https://code.visualstudio.com/)

Create a new python environment
````
conda create -p myenv python==3.7 -y
````
Activate it and install required packages
````
pip install -r requirements.txt
````

