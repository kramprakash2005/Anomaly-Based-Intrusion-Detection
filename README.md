# Comparative Analysis of Supervised Machine Learning Algorithms for Anomaly-Based Intrusion Detection

## Abstract
This repository presents a comparative analysis of various supervised machine learning algorithms for anomaly-based intrusion detection using the KDD Cup 1999 dataset. The study evaluates the performance of models including Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Naive Bayes, and XGBoost. Performance metrics such as Accuracy, F1 Score, Precision, Recall, and AUC are utilized for model evaluation.


## Introduction
Anomaly-based intrusion detection is crucial in network security. This research aims to compare the effectiveness of several supervised machine learning algorithms in detecting network intrusions. The KDD Cup 1999 dataset is used to train and evaluate the models.


## Methodology
### Dataset
The KDD Cup 1999 dataset is used, comprising various network traffic data labeled as normal or attack.

### Model Selection
The following supervised machine learning models are implemented:
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Naive Bayes
- XGBoost

### Evaluation Metrics
The models are evaluated based on Accuracy, F1 Score, Precision, Recall, and AUC.

## Experimental Setup
The dataset is split into training and testing sets. Models are trained and hyperparameters are tuned for optimal performance.

## Results
### Model Performance Results
The performance results for each model are as follows:

| Model                        | Accuracy | F1 Score | Precision | Recall  | AUC      |
|------------------------------|----------|----------|-----------|---------|----------|
| Logistic Regression           | 0.9716   | 0.9695   | 0.9773    | 0.9617  | 0.9963   |
| Decision Tree                 | 0.9985   | 0.9984   | 0.9988    | 0.9981  | 0.9985   |
| Random Forest                 | 0.9989   | 0.9988   | 0.9995    | 0.9981  | 0.9999   |
| K-Nearest Neighbors (KNN)    | 0.9959   | 0.9956   | 0.9962    | 0.9951  | 0.9994   |
| Support Vector Machine (SVM)  | 0.9923   | 0.9918   | 0.9903    | 0.9932  | 0.9991   |
| Naive Bayes                  | 0.8656   | 0.8332   | 0.9958    | 0.7163  | 0.9806   |
| XGBoost                      | 0.9992   | 0.9991   | 0.9994    | 0.9989  | 0.9999   |

### Model Performance Comparison (Sorted by Aggregate Score)
The models sorted by their aggregate score are as follows:

| Model                        | Aggregate Score |
|------------------------------|------------------|
| XGBoost                      | 0.9992           |
| Random Forest                | 0.9989           |
| Decision Tree                | 0.9985           |
| K-Nearest Neighbors (KNN)    | 0.9961           |
| Support Vector Machine (SVM) | 0.9926           |
| Logistic Regression           | 0.9727           |
| Naive Bayes                  | 0.8652           |


## Discussion
The results indicate that XGBoost and Random Forest outperform other models in terms of accuracy and F1 score. The findings contribute valuable insights for practitioners in the field of network security.

## Conclusion
This study provides a comprehensive comparison of supervised machine learning algorithms for anomaly detection. Future work may explore hybrid models or deep learning approaches for enhanced performance.



## License
This project is licensed under the MIT License.

