# COVID-19 Detection Model Readme

## Introduction

This project focuses on building and evaluating machine learning models for the detection of COVID-19 using a dataset containing various clinical and demographic features. The goal was to create a robust classifier capable of distinguishing between individuals who are likely to have COVID-19 and those who are not. The models were trained, evaluated, and compared based on key performance metrics such as accuracy, true positive ratio, true negative ratio, false positive ratio, and false negative ratio.

The approach in this notebook involves data preprocessing, feature selection, model training, and evaluation, with special attention to handling issues such as missing data, class imbalance, and overfitting.

## Table of Contents

1. [Objective and Dataset Overview](#objective-and-dataset-overview)
2. [Data Preprocessing and Exploration](#data-preprocessing-and-exploration)
3. [Model Building and Evaluation](#model-building-and-evaluation)
4. [Findings and Key Insights](#findings-and-key-insights)
5. [Conclusion](#conclusion)

---

## Objective and Dataset Overview

The primary objective of this project is to develop a model capable of accurately predicting whether a patient has COVID-19 based on their medical records. The dataset used for this analysis includes clinical features such as age, symptoms, comorbidities, and test results, and is labeled with a binary classification indicating whether the individual is positive for COVID-19.

### Key Steps:
1. **Data Preprocessing**: Cleaning and preparing the data for model training.
2. **Feature Engineering**: Creating meaningful features that enhance model performance.
3. **Model Training**: Training multiple machine learning classifiers and evaluating their performance.
4. **Evaluation and Comparison**: Comparing the models' performance using accuracy and other evaluation metrics.

---

## Data Preprocessing and Exploration

### Handling Missing Values

One of the first issues identified during data exploration was the presence of missing values. These missing values, if not addressed properly, could lead to skewed results and affect model performance. We used the `df.corr()` function to inspect correlations in the dataset and discovered that some correlations were distorted due to missing values. For instance, certain features that should have been highly correlated were not, because the rows with missing data were not handled properly.

### Data Cleaning and Imputation

To address this issue, we used several strategies for handling missing values:
- **Imputation**: Missing values for continuous features were imputed using the mean or median, depending on the feature distribution.
- **Dropping Rows**: Rows with missing values for categorical variables, particularly the target variable (COVID status), were dropped.

### Stratified Sampling

We also used stratified sampling to ensure that the distribution of COVID-positive and COVID-negative individuals in the training and test sets was representative of the overall dataset. This approach was crucial for avoiding biases, especially since the dataset might have had an imbalance between the two classes.

---

## Model Building and Evaluation

### Model Selection

We evaluated several machine learning classifiers, ranging from traditional algorithms like **Logistic Regression** and **Naive Bayes** to more complex ensemble methods like **Random Forest** and **Extra Trees**. All models were evaluated on their ability to predict COVID status using several key metrics.

### Evaluation Metrics

The performance of each model was evaluated using the following metrics:

- **Accuracy**: The proportion of correct predictions (both positive and negative).
- **True Positive Ratio (TPR)**: The proportion of actual positives correctly identified by the model.
- **True Negative Ratio (TNR)**: The proportion of actual negatives correctly identified by the model.
- **False Positive Ratio (FPR)**: The proportion of actual negatives incorrectly classified as positives.
- **False Negative Ratio (FNR)**: The proportion of actual positives incorrectly classified as negatives.

### Results

#### First Set of Results

| Algorithm                 | Accuracy | True Positive Ratio | True Negative Ratio | False Positive Ratio | False Negative Ratio |
|---------------------------|----------|---------------------|---------------------|----------------------|----------------------|
| Random Forest             | 91.17%   | 58.72%              | 96.35%              | 3.65%                | 41.28%               |
| Decision Tree             | 90.95%   | 56.30%              | 96.48%              | 3.52%                | 43.70%               |
| Support Vector Machine    | 90.59%   | 55.65%              | 96.16%              | 3.84%                | 44.35%               |
| K=7 Nearest Neighbours    | 90.24%   | 57.64%              | 95.44%              | 4.56%                | 42.36%               |
| Logistic Regression       | 90.10%   | 56.80%              | 95.41%              | 4.59%                | 43.20%               |
| K=5 Nearest Neighbours    | 90.00%   | 57.03%              | 95.26%              | 4.74%                | 42.97%               |
| K=3 Nearest Neighbours    | 89.48%   | 58.29%              | 94.46%              | 5.54%                | 41.71%               |
| K=1 Nearest Neighbours    | 88.65%   | 55.81%              | 93.89%              | 6.11%                | 44.19%               |
| Naive Bayes               | 83.31%   | 87.31%              | 82.67%              | 17.33%               | 12.69%               |

#### Second Set of Results

| Algorithm                 | Accuracy | True Positive Ratio | True Negative Ratio | False Positive Ratio | False Negative Ratio |
|---------------------------|----------|---------------------|---------------------|----------------------|----------------------|
| Extra Trees               | 91.48%   | 57.75%              | 96.86%              | 3.14%                | 42.25%               |
| Random Forest             | 91.32%   | 51.12%              | 97.73%              | 2.27%                | 48.88%               |
| SVC                       | 90.57%   | 55.50%              | 96.16%              | 3.84%                | 44.50%               |

### Model Analysis and Key Findings

- **Random Forest** consistently performed well across both the accuracy and true negative ratio, making it a robust choice for distinguishing between COVID-positive and COVID-negative individuals.
- **Naive Bayes** had the highest true positive ratio, but it also exhibited the highest false negative ratio, indicating that while it was good at identifying true positives, it was prone to making false negatives.
- **Extra Trees** outperformed Random Forest slightly in terms of accuracy in the second set of results, but the true positive ratio was slightly lower.
- The **Support Vector Machine** model, while strong in true negative prediction, had a relatively low true positive ratio and was prone to false negatives.

---

## Findings and Key Insights

- **Missing Data and Correlations**: Missing values in the dataset led to unusual correlations. The `df.corr()` function helped us identify these issues, which were addressed by proper data imputation and cleaning.
- **Model Comparison**: Among the models tested, **Random Forest** and **Extra Trees** stood out for their high accuracy and robust performance. However, models like **Naive Bayes** provided valuable insights into the true positive ratio, which can be useful depending on the business goals (minimizing false negatives).
- **Stratified Sampling**: The use of stratified sampling ensured that the model’s performance metrics were not biased by class imbalance, allowing us to train on a representative subset of the population.

---

## Conclusion

This analysis demonstrates the effectiveness of machine learning models in detecting COVID-19 from clinical data. The key takeaway is that while ensemble methods like **Random Forest** and **Extra Trees** are highly accurate, there are trade-offs in terms of true positive and false negative ratios. Depending on the application, focusing on minimizing false negatives may be more important, which could favor models like **Naive Bayes**.

Future work could involve refining the models further, exploring deep learning techniques, or incorporating additional features (such as genomic data or imaging data) to improve accuracy. Additionally, fine-tuning hyperparameters for each algorithm could potentially yield better results.

This project not only highlights the technical ability to develop predictive models but also showcases the importance of data preprocessing and understanding the nuances in the data to improve model outcomes.