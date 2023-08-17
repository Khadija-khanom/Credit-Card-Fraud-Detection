# Credit-Card-Fraud-Detection

# Data Visualization
![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/0dd94017-5b26-42a5-932d-5dfe6aab099e)
Distribution ![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/31670df0-bd58-4bcd-8ec2-a7c710395a9a)
Amount Distribution ![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/d87fab27-9f7b-41aa-b787-ca18a6b34885)
Correlation Matrix ![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/f4876e4d-6fea-417d-ad1e-25130f2c498b)
Pairplot of selected features ![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/b30b86bb-cff1-4edb-b88d-997290fea49a)
# Implementation of Machine learning Models
This section demonstrates the process of building and evaluating several machine learning models for credit card fraud detection using the provided dataset. It covers the Random Forest, Decision Tree, Logistic Regression, and K-Nearest Neighbors models. Let's break down the structure and the process of building these models:

Model Initialization: For each model (Random Forest, Decision Tree, Logistic Regression, and K-Nearest Neighbors), the code initializes a corresponding model object with specified hyperparameters.
Training the Models: For each model, the resampled training data (X_train_resampled, y_train_resampled) obtained from applying SMOTE is used to train the model using the fit method.
Making Predictions: After training each model, predictions are made on the test set (X_test) using the predict method, and the predicted labels are stored in y_rf_pred, y_dt_pred, y_lr_pred, and y_knn_pred.
Evaluating the Models:For each model, various evaluation metrics are computed and printed:
• Accuracy: Calculated using accuracy_score by comparing the predicted labels (y_rf_pred, y_dt_pred, etc.) with the actual test labels (y_test).

• Confusion Matrix: Generated using confusion_matrix to show the counts of true positive, true negative, false positive, and false negative predictions.

• Classification Report: Generated using classification_report to display metrics such as precision, recall, F1-score, and support for both classes.

Model Comparison: • The printed evaluation metrics (accuracy, confusion matrix, classification report) for each model provide a comparison of their performance on the test set.
Model Building Process:

The process starts by initializing each model with specific hyperparameters.
The models are trained using the resampled training data created through SMOTE, which addresses class imbalance.
Predictions are made on the test set for each model.
Model performance is evaluated using accuracy, confusion matrix, and classification report metrics.
The results for all models are compared to understand which one performs best for the task of credit card fraud detection.
[ ]
# Predicted Result Evaluation
Predicted Result Evaluation
Describing and explaining the predicted results of the models based on the provided evaluation metrics:

Random Forest Model:

• Accuracy: 99.81%

• Precision (fraudulent transactions): 48%

• Recall (fraudulent transactions): 89%

• F1-score (fraudulent transactions): 62%

Explanation: The Random Forest model achieves high accuracy and is able to detect a significant portion of fraudulent transactions (high recall). However, the precision is relatively low, indicating that some of the predicted fraudulent transactions are false positives. This trade-off between precision and recall is reflected in the F1-score, which balances both metrics. Overall, the model performs well in detecting fraudulent transactions, but there is room for improvement in reducing false positives.

Decision Tree Model:

• Accuracy: 98.43%

• Precision (fraudulent transactions): 9%

• Recall (fraudulent transactions): 84%

• F1-score (fraudulent transactions): 15%

Explanation: The Decision Tree model achieves a high accuracy, but the precision is very low, indicating a high rate of false positives. The recall is relatively high, showing that the model detects a significant portion of fraudulent transactions, but the trade-off between precision and recall results in a low F1-score. This suggests that the model is missing some fraudulent transactions while producing many false positives.

Logistic Regression Model: • Accuracy: 97.46%

• Precision (fraudulent transactions): 6%

• Recall (fraudulent transactions): 92%

• F1-score (fraudulent transactions): 11%

Explanation: The Logistic Regression model achieves good accuracy and high recall, indicating it's able to capture a large portion of fraudulent transactions. However, the precision is very low, meaning there are many false positives. The F1-score is also low due to the imbalance between precision and recall. This model shows promise in identifying fraudulent transactions but needs improvement in reducing false positives.

K-Nearest Neighbors Model:

• Accuracy: 99.82%

• Precision (fraudulent transactions): 48%

• Recall (fraudulent transactions): 87%

• F1-score (fraudulent transactions): 62%

Explanation: The K-Nearest Neighbors model achieves high accuracy and is effective at detecting fraudulent transactions, with a balance between precision and recall. Similar to the Random Forest model, there's a trade-off between precision and recall, as indicated by the F1-score. The model performs well overall, but there's still room for improvement in terms of reducing false positives.

Summary: All models achieve high accuracy, but there's a common trade-off between precision and recall, leading to moderate F1-scores. This suggests that the models have the potential to detect a significant portion of fraudulent transactions, but they also produce false positives. Further model tuning and potentially exploring ensemble methods or more sophisticated algorithms might help in achieving a better balance between precision and recall, ultimately leading to improved performance in credit card fraud detection.

# Learning Curve of Machine Learning Models 
![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/4528c6f4-c3bd-4814-b14d-102f8bc0d7af)

