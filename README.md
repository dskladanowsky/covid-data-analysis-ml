# COVID Data Analysis and Model Training

## Overview

This project involves the analysis of a COVID-19 dataset provided by the Mexican government. The dataset includes a vast amount of anonymized patient information, and the analysis focuses on exploring the data, preprocessing it, and training various machine learning models for the classification of patient outcomes.

## Dataset

The dataset contains 21 unique features and information on 1,048,575 unique patients. Features include patient characteristics, test results, and medical history.

## Data Analysis and Visualization

The project includes exploratory data analysis (EDA) and visualization of the dataset. Key visualizations include count plots, correlation matrices, and the distribution of various attributes.

## Data Preprocessing

Several preprocessing steps were performed to prepare the dataset for model training. This includes handling missing data, removing unnecessary columns, discretizing the 'AGE' attribute, and creating a new decision attribute 'DECEASED' based on the 'DATE_DIED' column.

## Machine Learning Models

The project involves training various machine learning models for predicting patient outcomes. The following models were trained and evaluated:

- Gaussian Naive Bayes
- Logistic Regression
- K-Nearest Neighbors (K=1, K=3, K=5, K=7)
- Support Vector Machine (LinearSVC)
- Decision Tree
- Random Forest

## Model Evaluation

The performance of each trained model is evaluated using metrics such as accuracy, true positive ratio, true negative ratio, false positive ratio, false negative ratio, and a detailed classification report.

## Instructions for Running the Code

1. Install the required libraries.
2. Run the Jupyter notebook `Covid_Data_Analysis_And_Model_Training.ipynb` step by step.

## Conclusion

The project provides insights into the COVID-19 dataset, preprocesses the data, and trains various machine learning models for outcome prediction. The README serves as a guide for understanding and replicating the analysis.

## Acknowledgments

- The dataset was provided by the Mexican government.

Feel free to reach out for questions!
