# **Credit Card Fraud Detection using Sampling Techniques**

## **Overview**
This project demonstrates the application of sampling techniques to handle imbalanced datasets, particularly in the context of credit card fraud detection. Various sampling methods are employed to balance the dataset, followed by training multiple machine learning models to evaluate their performance across these techniques.

---

## **Problem Statement**
Credit card fraud detection datasets are often highly imbalanced, where fraudulent transactions make up a tiny fraction of the data. This imbalance poses a challenge for machine learning models, as they tend to be biased toward the majority class. To address this issue, this project focuses on:
1. Balancing the dataset using oversampling techniques.
2. Applying different sampling methods (e.g., random, stratified, systematic, cluster sampling).
3. Evaluating machine learning models' performance using these sampling techniques.

---

## **Steps Implemented**
### **1. Dataset Preprocessing**
- **Dataset**: [Credit Card Data](https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv)
- The dataset contains features and a target column (`Target`) representing whether a transaction is fraudulent (1) or not (0).
- Handled missing values and performed initial exploration to understand the data distribution.

### **2. Balancing the Dataset**
- **Technique Used**: Oversampling the minority class (fraudulent transactions) to match the majority class using the `resample()` method from `sklearn.utils`.
- Ensured a balanced dataset to avoid model bias toward the majority class.

### **3. Sampling Techniques**
Five sampling techniques were implemented:
1. **Random Sampling**: Randomly selects a subset of rows from the dataset.
2. **Stratified Sampling**: Ensures the target class distribution is maintained in the samples.
3. **Systematic Sampling**: Selects every nth row from the dataset.
4. **Cluster Sampling**: Groups the data based on a feature (e.g., `'Amount'`) and selects specific clusters for training.
5. **Full Dataset**: Includes the entire balanced dataset for comparison.

### **4. Machine Learning Models**
Two machine learning models were trained on each sample:
- Logistic Regression
- Decision Tree Classifier

### **5. Model Evaluation**
- The models were evaluated on the same samples used for training to understand the impact of sampling techniques.
- **Metric Used**: Accuracy.

### **6. Results and Insights**
- Results were compared to determine the best sampling technique for each model.
- The sampling technique and model combinations yielding the highest accuracy were identified.

---

## **Project Structure**
