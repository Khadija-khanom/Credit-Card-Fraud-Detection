# Comparative Analysis of Machine Learning and Deep Learning Algorithms for Credit Card Fraud Detection on 1st Dataset 
![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/1d1ab48c-6150-4def-9fe8-ef9a31d787ad)
(credit card transactions made by European cardholders in September 2013)

The prevalence of credit card fraud continues to pose significant financial and security risks for individuals, businesses, and financial institutions. Traditional rule-based systems are often insufficient in detecting increasingly sophisticated fraudulent activities. To address this challenge, the aim of this study is to develop and evaluate machine learning and deep learning models for accurate and efficient credit card fraud detection.

**Background**: The dataset used in this study contains credit card transactions made by European cardholders in September 2013. It includes features resulting from Principal Component Analysis (PCA) transformation, which ensures the confidentiality of the original data. The dataset is highly imbalanced, with a very small proportion of fraudulent transactions compared to legitimate ones.

**Data Source and Timeframe**: The dataset consists of credit card transactions that occurred in September 2013. The transactions were made by European cardholders.

**Transaction Details**: The dataset covers transactions over a span of two days. Out of the 284,807 transactions in the dataset, 492 of them are labelled as fraudulent (positive class). This means that the vast majority (99.828%) of the transactions are legitimate (negative class).

**Data Imbalance**: The dataset is highly imbalanced due to the small number of fraudulent transactions compared to legitimate ones. This imbalance is common in real-world fraud detection scenarios, as fraudulent transactions are typically a rare occurrence.

**Features**: The dataset contains only numerical input features. Most of these features (V1 to V28) are the result of applying Principal Component Analysis (PCA) transformation to the original features. Unfortunately, the original features and additional background information are not provided due to confidentiality issues. The 'Time' feature represents the time elapsed between a transaction and the first transaction in the dataset. The 'Amount' feature represents the transaction amount.

**Response Variable**: The 'Class' feature is the response variable that indicates whether a transaction is fraudulent (1) or legitimate (0).

**Objective**:The primary objective of this study is to build and evaluate predictive models capable of identifying fraudulent credit card transactions with high accuracy, while minimizing false positives and false negatives. This involves the application of various machine learning and deep learning techniques to effectively handle the imbalanced nature of the dataset and capture intricate patterns associated with fraudulent activities.
Table of Contents
=================

  * [Data Visualization]
  * [Implementation of Machine learning Models]
  * [Installation]
        * [OR using Pathogen:](#or-using-pathogen)
        * [OR using Vundle:](#or-using-vundle)
  * [License](#license)
# Data Visualization
![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/0dd94017-5b26-42a5-932d-5dfe6aab099e)
Distribution ![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/31670df0-bd58-4bcd-8ec2-a7c710395a9a)
Amount Distribution ![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/d87fab27-9f7b-41aa-b787-ca18a6b34885)
Correlation Matrix ![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/f4876e4d-6fea-417d-ad1e-25130f2c498b)
Pairplot of selected features ![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/b30b86bb-cff1-4edb-b88d-997290fea49a)
Box plots to visualize outliers ![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/57569c19-2048-4b3b-9197-19ad8836359c)

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
![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/15854ad1-a692-45be-bf7c-01ae13d959ab)
![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/981d724c-a69c-4c62-8ab0-d3ffd47e7954)


# Implementation Of Deep learning model 
Discussing the structure of the CNN and RNN models and how they were constructed:

Convolutional Neural Network (CNN): The CNN is a type of deep learning model well-suited for processing grid-like data such as images and sequences. In your code, you've built a simple 1D CNN for fraud detection:

Model Structure:

The CNN model consists of multiple layers:

Conv1D layer: This layer applies convolutional filters to the input data. It has 64 filters with a kernel size of 3 and uses the ReLU activation function.

MaxPooling1D layer: This layer performs max pooling to reduce the spatial dimensions of the data.

Flatten layer: This layer converts the pooled feature maps into a 1D vector.

Dense layers: There are two dense layers with 128 and 1 neurons respectively. The first dense layer uses the ReLU activation function, while the last dense layer uses the sigmoid activation function for binary classification.

Compilation:

The model is compiled using the Adam optimizer and binary cross-entropy loss, which is suitable for binary classification tasks. The accuracy metric is also specified for evaluation. Training:

The model is trained using the training data (X_train_resampled and y_train_resampled) reshaped to fit the input shape of the model. The model is trained for 10 epochs with a batch size of 64.

Evaluation:

After training, the model's predictions are obtained for the test data (X_test) and then thresholded using 0.5 to convert them into binary predictions. The accuracy, confusion matrix, and classification report are printed to evaluate the model's performance. Recurrent Neural Network (RNN): RNNs are well-suited for sequential data, where the order of elements matters. Here's how you've constructed the RNN:

Model Structure:

The RNN model is built using the Long Short-Term Memory (LSTM) cell, a type of recurrent unit that can capture long-range dependencies in sequences.

LSTM layer: This layer has 64 LSTM units with the ReLU activation function and takes the input shape of the resampled training data. Compilation and Training:

Similar to the CNN, the RNN is compiled using the Adam optimizer and binary cross-entropy loss. The model is trained using the reshaped training data for 10 epochs with a batch size of 64.

Evaluation:

After training, the model's predictions are obtained for the test data and thresholded using 0.5 for binary predictions. The accuracy, confusion matrix, and classification report are printed to evaluate the model's performance. Building the Models:

Both models are constructed using the Keras Sequential API, allowing you to stack layers sequentially. The choice of layers, activation functions, and optimizer is based on experimentation, known best practices, and the nature of the data (sequential for RNN).

Summary: Both a CNN and an RNN were built and trained for credit card fraud detection. CNNs are effective for identifying patterns in sequences, while RNNs are suitable for sequential data. The performance of these models can vary based on their architecture, hyperparameters, and the characteristics of the dataset. Experimentation and tuning may be necessary to optimize their performance further.
# Predicted Result Evaluation 
CNN Model:

The CNN model achieved an accuracy of approximately 99.90%.

The confusion matrix indicates that out of 56,864 non-fraudulent transactions, 56,825 were correctly predicted as non-fraudulent (true negatives). Additionally, out of 98 fraudulent transactions, 82 were correctly predicted as fraudulent (true positives).

The classification report provides further insights into the precision, recall, and F1-score. The model achieved a precision of 0.68 for detecting fraudulent transactions, indicating that when it predicts a transaction as fraudulent, it's correct 68% of the time. The recall (true positive rate) is 0.84, which suggests that the model is able to identify 84% of the actual fraudulent transactions.

RNN Model:

The RNN model achieved an accuracy of approximately 99.53%.

The confusion matrix indicates that out of 56,864 non-fraudulent transactions, 56,609 were correctly predicted as non-fraudulent (true negatives). Moreover, out of 98 fraudulent transactions, 86 were correctly predicted as fraudulent (true positives).

The classification report highlights that the model has a lower precision of 0.25 for detecting fraudulent transactions, indicating that its predictions for fraud may have a higher false positive rate. However, the recall is 0.88, suggesting that the model is able to identify 88% of the actual fraudulent transactions.

Summary:

The CNN model outperforms the RNN model in terms of accuracy, precision, and recall for fraud detection.

The CNN model has a higher precision and recall, indicating that it's better at correctly identifying both non-fraudulent and fraudulent transactions.

The RNN model has a lower precision and higher recall, implying that it may produce more false positives but captures a larger proportion of actual fraud cases.

Overall, based on the provided results, the CNN model appears to be the better performer between the two for this specific fraud detection task.

# Learning Curve of Deep Learning Model
![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/660ff616-bba2-4d4f-a496-6bbab285cd38)
# comparative analysis Between Machine learning models and Deep learning models
Among the models we have trained, the CNN (Convolutional Neural Network) model appears to provide the best results for credit card fraud detection. Let's explain why the CNN model performed the best in detail:

Model Performances and Explanation:

Random Forest:

Accuracy: 99.81%
F1-score (Fraud Class): 0.62
Random Forest achieved high accuracy but has a relatively lower F1-score for the fraud class. It might be performing well on non-fraudulent cases but is not as effective in identifying fraud cases.
Decision Tree:

Accuracy: 98.43%
F1-score (Fraud Class): 0.15
The Decision Tree model has good accuracy, but its low F1-score for the fraud class indicates it might have a higher number of false negatives, classifying some fraud cases as non-fraud.
Logistic Regression:

Accuracy: 97.46%
F1-score (Fraud Class): 0.11
While Logistic Regression has high accuracy, its low F1-score for the fraud class suggests it struggles to effectively detect fraud cases, leading to a high number of false negatives.
K-Nearest Neighbors:

Accuracy: 99.82%
F1-score (Fraud Class): 0.62
K-Nearest Neighbors achieved high accuracy but has an F1-score similar to the Random Forest model. It might have challenges in effectively capturing complex patterns associated with fraud cases.
CNN:

Accuracy: 99.90%
F1-score (Fraud Class): 0.75
The CNN model achieved the highest accuracy and F1-score for the fraud class. It excels at capturing complex patterns and relationships in sequential data, making it particularly suitable for fraud detection where intricate patterns are crucial.
RNN:

Accuracy: 99.53%
F1-score (Fraud Class): 0.39
The RNN model has high accuracy, but its F1-score for the fraud class is lower compared to the CNN model. This suggests that while it's identifying fraud cases, it might also be generating a higher number of false positives.
Why the CNN Model Performed Best:

Feature Extraction: CNNs are effective at learning relevant features from sequential data. In the case of credit card transactions, CNNs can capture hidden patterns that other models might miss.
Pattern Recognition: Fraudulent activities often involve intricate patterns that can be better captured by CNNs' ability to identify local patterns and hierarchies.
Complex Relationships: CNNs can identify relationships between different features and their impact on the prediction, which is important for capturing sophisticated fraud patterns.
Conclusion:

Considering the high accuracy, F1-score, and the ability to capture complex patterns, the CNN model stands out as the best performer for credit card fraud detection. Its superior performance in correctly classifying fraud cases while maintaining a good balance between precision and recall makes it the recommended choice for this specific task.
