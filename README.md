# Comparative Analysis of Machine Learning and Deep Learning Algorithms for Credit Card Fraud Detection on 1st Dataset 
![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/1d1ab48c-6150-4def-9fe8-ef9a31d787ad)
(credit card transactions made by European cardholders in September 2013)

**About Dataset**: The dataset used in this study contains credit card transactions made by European cardholders in September 2013. It includes features resulting from Principal Component Analysis (PCA) transformation, which ensures the confidentiality of the original data. The dataset is highly imbalanced, with a very small proportion of fraudulent transactions compared to legitimate ones.

**Data Source and Timeframe**: The dataset consists of credit card transactions that occurred in September 2013. The transactions were made by European cardholders.

**Transaction Details**: The dataset covers transactions over a span of two days. Out of the 284,807 transactions in the dataset, 492 of them are labelled as fraudulent (positive class). This means that the vast majority (99.828%) of the transactions are legitimate (negative class).

**Data Imbalance**: The dataset is highly imbalanced due to the small number of fraudulent transactions compared to legitimate ones. This imbalance is common in real-world fraud detection scenarios, as fraudulent transactions are typically a rare occurrence.

**Features**: The dataset contains only numerical input features. Most of these features (V1 to V28) are the result of applying Principal Component Analysis (PCA) transformation to the original features. Unfortunately, the original features and additional background information are not provided due to confidentiality issues. The 'Time' feature represents the time elapsed between a transaction and the first transaction in the dataset. The 'Amount' feature represents the transaction amount.

**Response Variable**: The 'Class' feature is the response variable that indicates whether a transaction is fraudulent (1) or legitimate (0).

Table of Contents
=================

[Data Visualization](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/blob/main/README.md#data-visualization)

[Implementation of Machine learning Models](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/blob/main/README.md#implementation-of-machine-learning-models)

* [Model Building Process](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/blob/main/README.md#model-building-process)

[Evaluating the Performance of Machine Learning Models]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/blob/main/README.md#evaluating-the-performance-of-machine-learning-models)

* [Random Forest]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/blob/main/README.md#random-forest)

* [Decision Tree]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/blob/main/README.md#decision-tree)
   
* [Logistic Regression]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/blob/main/README.md#logistic-regression)
  
* [K-Nearest Neighbours]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/blob/main/README.md#k-nearest-neighbors)
  
* [Best Performing Model]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/blob/main/README.md#best-performing-model)
  
* [Learning Curve of Machine Learning Models]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/blob/main/README.md#learning-curve-of-machine-learning-models)
  

[Implementation Of Deep learning model]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/blob/main/README.md#implementation-of-deep-learning-model)

* [Convolutional Neural Network (CNN)]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/blob/main/README.md#convolutional-neural-network-cnn)
  
  - [Model Structure]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/blob/main/README.md#model-structure)

  - [Compilation]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/blob/main/README.md#compilation)
  
  - [Evaluation]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/blob/main/README.md#evaluation)
    
* [Recurrent Neural Network (RNN)]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/blob/main/README.md#recurrent-neural-network-rnn)
  
  - [Model Structure]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection#model-structure-1)
    
  - [Compilation and Training]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection#compilation-and-training-1)
    
  - [Evaluation]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection#evaluation-1)
    
[Evaluating the performance of deep learning models]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection#evaluating-the-performance-of-deep-learning-models)

* [CNN Model]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection#cnn-model)
  
* [RNN Model]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection#rnn-model)
  
* [Learning Curve of Deep Learning Model]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection#learning-curve-of-deep-learning-model)
  
[comparative analysis Between Machine learning models and Deep learning models]( https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection#comparative-analysis-between-machine-learning-models-and-deep-learning-models)

 
## Data Visualization
![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/0dd94017-5b26-42a5-932d-5dfe6aab099e)
Distribution ![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/31670df0-bd58-4bcd-8ec2-a7c710395a9a)
Amount Distribution ![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/d87fab27-9f7b-41aa-b787-ca18a6b34885)
Correlation Matrix ![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/f4876e4d-6fea-417d-ad1e-25130f2c498b)
Pairplot of selected features ![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/b30b86bb-cff1-4edb-b88d-997290fea49a)
Box plots to visualize outliers ![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/57569c19-2048-4b3b-9197-19ad8836359c)

## Implementation of Machine learning Models
This section demonstrates the process of building and evaluating several machine learning models for credit card fraud detection using the provided dataset. It covers the Random Forest, Decision Tree, Logistic Regression, and K-Nearest Neighbors models. Let's break down the structure and the process of building these models:

**Model Initialization**: For each model (Random Forest, Decision Tree, Logistic Regression, and K-Nearest Neighbors), the code initializes a corresponding model object with specified hyperparameters.

**Training the Models**: For each model, the resampled training data (X_train_resampled, y_train_resampled) obtained from applying SMOTE is used to train the model using the fit method.

**Making Predictions**: After training each model, predictions are made on the test set (X_test) using the predict method, and the predicted labels are stored in y_rf_pred, y_dt_pred, y_lr_pred, and y_knn_pred.

Evaluating the Models:For each model, various evaluation metrics are computed and printed:
• **Accuracy**: Calculated using accuracy_score by comparing the predicted labels (y_rf_pred, y_dt_pred, etc.) with the actual test labels (y_test).

• **Confusion Matrix**: Generated using confusion_matrix to show the counts of true positive, true negative, false positive, and false negative predictions.

• **Classification Report**: Generated using classification_report to display metrics such as precision, recall, F1-score, and support for both classes.

**Model Comparison**: The printed evaluation metrics (accuracy, confusion matrix, classification report) for each model provide a comparison of their performance on the test set.

### Model Building Process:

1. The process starts by initializing each model with specific hyperparameters.
2. The models are trained using the resampled training data created through SMOTE, which addresses class imbalance.
3. Predictions are made on the test set for each model.
4. Model performance is evaluated using accuracy, confusion matrix, and classification report metrics.
5. The results for all models are compared to understand which one performs best for the task of credit card fraud detection.

## Evaluating the Performance of Machine Learning Models
Here's the table with accuracy, precision, recall, and F1-score for both classes 0 and 1, followed by the analysis of the best-performing model:

![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/a40820d7-217e-469b-84aa-6556aa750e8d)

Now, analyzing the performance of these models and determine which one performed best:

### Random Forest:

   * Highest precision for class 0 (non-fraudulent transactions), indicating that it accurately predicts a high percentage of actual non-fraudulent cases.
   * High precision and recall for class 1, making it effective at both identifying actual fraudulent transactions and non-fraudulent transactions.

### Decision Tree:

   * High precision for class 0 and recall for class 1.
   * Very low precision for class 1, indicating a high number of false positives
   * High recall for class 1, suggesting it identifies a good portion of actual fraudulent transactions, but at the cost of more false positives.

### Logistic Regression:

   * High precision and recall for class 0.
   * Very low precision for class 1, leading to many false positives.
   * Very high recall for class 1, indicating that it captures most actual fraudulent transactions.

### K-Nearest Neighbors:

   * Highest precision for class 0.
   * Similar precision, recall, and F1-score for class 1 as Random Forest, indicating its effectiveness in identifying fraudulent transactions.

### Best Performing Model: 
Taking into consideration accuracy, precision, recall, and F1-score for both classes, the** Random Forest model** emerges as the best performer. It achieves an outstanding balance between precision and recall for both classes, indicating its robustness in identifying both fraudulent and non-fraudulent transactions accurately. The ensemble nature of Random Forest contributes to its ability to generalize well and manage overfitting, making it a suitable choice for fraud detection tasks.

### Learning Curve of Machine Learning Models
![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/4528c6f4-c3bd-4814-b14d-102f8bc0d7af)
![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/15854ad1-a692-45be-bf7c-01ae13d959ab)
![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/981d724c-a69c-4c62-8ab0-d3ffd47e7954)


## Implementation Of Deep learning model 
Discussing the structure of the CNN and RNN models and how they were constructed:

### Convolutional Neural Network (CNN) 
The CNN is a type of deep learning model well-suited for processing grid-like data such as images and sequences. In your code, you've built a simple 1D CNN for fraud detection:

#### Model Structure

**The CNN model consists of multiple layers**

**Conv1D layer**: This layer applies convolutional filters to the input data. It has 64 filters with a kernel size of 3 and uses the ReLU activation function.

**MaxPooling1D layer**: This layer performs max pooling to reduce the spatial dimensions of the data.

**Flatten layer**: This layer converts the pooled feature maps into a 1D vector.

***Dense layers**: There are two dense layers with 128 and 1 neurons respectively. The first dense layer uses the ReLU activation function, while the last dense layer uses the sigmoid activation function for binary classification.

#### Compilation and Training:

The model is compiled using the Adam optimizer and binary cross-entropy loss, which is suitable for binary classification tasks. The accuracy metric is also specified for evaluation.The model is trained using the training data (X_train_resampled and y_train_resampled) reshaped to fit the input shape of the model. The model is trained for 10 epochs with a batch size of 64.

#### Evaluation:

After training, the model's predictions are obtained for the test data (X_test) and then thresholded using 0.5 to convert them into binary predictions. The accuracy, confusion matrix, and classification report are printed to evaluate the model's performance.
### Recurrent Neural Network (RNN)
RNNs are well-suited for sequential data, where the order of elements matters. The RNN model building process is presented below:

#### Model Structure

The RNN model is built using the Long Short-Term Memory (LSTM) cell, a type of recurrent unit that can capture long-range dependencies in sequences.

**LSTM layer**: This layer has 64 LSTM units with the ReLU activation function and takes the input shape of the resampled training data. 

#### Compilation and Training
Similar to the CNN, the RNN is compiled using the Adam optimizer and binary cross-entropy loss. The model is trained using the reshaped training data for 10 epochs with a batch size of 64.

#### Evaluation

After training, the model's predictions are obtained for the test data and thresholded using 0.5 for binary predictions. The accuracy, confusion matrix, and classification report are printed to evaluate the model's performance. Building the Models:

Both models are constructed using the Keras Sequential API, allowing you to stack layers sequentially. The choice of layers, activation functions, and optimizer is based on experimentation, known best practices, and the nature of the data (sequential for RNN).

**Summary**: Both a CNN and an RNN were built and trained for credit card fraud detection. CNNs are effective for identifying patterns in sequences, while RNNs are suitable for sequential data. The performance of these models can vary based on their architecture, hyperparameters, and the characteristics of the dataset. Experimentation and tuning may be necessary to optimize their performance further.
## Evaluating the performance of deep learning models
### CNN Model

The CNN model achieved an accuracy of approximately 99.90%.

The confusion matrix indicates that out of 56,864 non-fraudulent transactions, 56,825 were correctly predicted as non-fraudulent (true negatives). Additionally, out of 98 fraudulent transactions, 82 were correctly predicted as fraudulent (true positives).

The classification report provides further insights into the precision, recall, and F1-score. The model achieved a precision of 0.68 for detecting fraudulent transactions, indicating that when it predicts a transaction as fraudulent, it's correct 68% of the time. The recall (true positive rate) is 0.84, which suggests that the model is able to identify 84% of the actual fraudulent transactions.

### RNN Model

The RNN model achieved an accuracy of approximately 99.53%.

The confusion matrix indicates that out of 56,864 non-fraudulent transactions, 56,609 were correctly predicted as non-fraudulent (true negatives). Moreover, out of 98 fraudulent transactions, 86 were correctly predicted as fraudulent (true positives).

The classification report highlights that the model has a lower precision of 0.25 for detecting fraudulent transactions, indicating that its predictions for fraud may have a higher false positive rate. However, the recall is 0.88, suggesting that the model is able to identify 88% of the actual fraudulent transactions.

**Summary**:

The CNN model outperforms the RNN model in terms of accuracy, precision, and recall for fraud detection.

The CNN model has a higher precision and recall, indicating that it's better at correctly identifying both non-fraudulent and fraudulent transactions.

The RNN model has a lower precision and higher recall, implying that it may produce more false positives but captures a larger proportion of actual fraud cases.

Overall, based on the provided results, the CNN model appears to be the better performer between the two for this specific fraud detection task.

### Learning Curve of Deep Learning Model
![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/660ff616-bba2-4d4f-a496-6bbab285cd38)
## comparative analysis Between Machine learning models and Deep learning models
The comparative analysis and the explanation of the best-performing model is presented below:
![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/c72d5988-5081-40c9-b38f-5a23e5271ed3)
![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/f8618ca3-55a0-4a26-85f9-b5ada31f95bb)

![image](https://github.com/Khadija-khanom/Credit-Card-Fraud-Detection/assets/138976722/f9ee65ce-9696-43fb-8ce6-fe945dcb3d81)



**Comparative Analysis:** Comparing the machine learning models (Random Forest, Decision Tree, Logistic Regression, K-Nearest Neighbors) with the deep learning models (CNN, RNN):

**Machine Learning Models:**

- Random Forest, Decision Tree, Logistic Regression, and K-Nearest Neighbors achieve high accuracy, but their performance varies significantly in terms of precision, recall, and F1-score for class 1.
- K-Nearest Neighbors and Random Forest perform relatively better due to their balanced precision and recall for class 1.

**Deep Learning Models:**

- CNN demonstrates superior performance with the highest accuracy, precision, recall, and F1-score for class 1 compared to other models.

- RNN also performs well with high recall for class 1, but its precision is comparatively lower, leading to more false positives.

**Best Performing Model:** Based on the metrics and analysis, the CNN model stands out as the best performer. It achieves the highest accuracy and maintains a balanced trade-off between precision and recall for class 1. This suggests that the CNN has a better ability to accurately detect both fraudulent and non-fraudulent transactions, likely due to its capacity to learn intricate spatial patterns in the data.

**Why the CNN Model Performed Best:**

**Feature Extraction:** CNNs are effective at learning relevant features from sequential data. In the case of credit card transactions, CNNs can capture hidden patterns that other models might miss.

**Pattern Recognition:** Fraudulent activities often involve intricate patterns that can be better captured by CNNs' ability to identify local patterns and hierarchies.
Complex Relationships: CNNs can identify relationships between different features and their impact on the prediction, which is important for capturing sophisticated fraud patterns.

**Conclusion:**

Considering the high accuracy, F1-score, and the ability to capture complex patterns, the CNN model stands out as the best performer for credit card fraud detection. Its superior performance in correctly classifying fraud cases while maintaining a good balance between precision and recall makes it the recommended choice for this specific task.


