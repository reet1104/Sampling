#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

url = "https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv"
data = pd.read_csv(url)

print(data.head())
print(data.info())


# In[7]:


from sklearn.utils import resample

# Separate majority and minority classes
majority = data[data['Class'] == 0]  # Replace 'Target' with your actual target column
minority = data[data['Class'] == 1]

# Oversample the minority class
minority_oversampled = resample(minority, 
                                replace=True, 
                                n_samples=len(majority), 
                                random_state=42)

# Combine the majority class with oversampled minority class
balanced_data = pd.concat([majority, minority_oversampled])

# Shuffle the dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)



# In[9]:


# Sampling sizes based on your formula or proportionally dividing the data
sample_sizes = [len(balanced_data) // 5] * 5  # Replace with formula or desired sizes

# Generate five samples
samples = [balanced_data.sample(n=size, random_state=i) for i, size in enumerate(sample_sizes)]



# In[11]:


from sklearn.model_selection import train_test_split

sampling_methods = []

# Random Sampling
random_sample = balanced_data.sample(n=100, random_state=42)  # Adjust size as needed
sampling_methods.append(random_sample)

# Stratified Sampling
X = balanced_data.drop(columns=['Class'])  # Replace 'Target' with your actual column name
y = balanced_data['Class']

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
stratified_sample = pd.concat([X_train, y_train], axis=1)
sampling_methods.append(stratified_sample)

# Add other techniques like cluster sampling, systematic sampling, etc.


# In[21]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Initialize classifiers
models = [
    LogisticRegression(max_iter=200, random_state=42),
    DecisionTreeClassifier(random_state=42)
]

# Prepare results dictionary
results = {}

# Sampling Techniques
sampling_methods = []

# Random Sampling
random_sample = balanced_data.sample(n=100, random_state=42)  # Adjust size
sampling_methods.append(random_sample)

# Stratified Sampling
X_train, _, y_train, _ = train_test_split(
    balanced_data.drop(columns=['Class']), 
    balanced_data['Class'], 
    test_size=0.5, 
    stratify=balanced_data['Class'], 
    random_state=42
)
stratified_sample = pd.concat([X_train, y_train], axis=1)
sampling_methods.append(stratified_sample)

# Systematic Sampling
systematic_sample = balanced_data.iloc[::10]  # Example: take every 10th row
sampling_methods.append(systematic_sample)

# Cluster Sampling (using a valid feature for clustering, e.g., 'Amount')
if 'Amount' in balanced_data.columns:
    cluster_sample = balanced_data[balanced_data['Amount'] < balanced_data['Amount'].quantile(0.5)]
else:
    print("Ensure the column used for clustering exists in your dataset.")
    cluster_sample = balanced_data.iloc[:100, :]  # Example fallback
sampling_methods.append(cluster_sample)

# Full Dataset (for comparison)
sampling_methods.append(balanced_data)

# Train models using each sampling method
for i, sample in enumerate(sampling_methods):
    X_sample = sample.drop(columns=['Class'])
    y_sample = sample['Class']
    
    for j, model in enumerate(models):
        # Fit the model
        model.fit(X_sample, y_sample)
        
        # Predict on the same sample (just for comparison purposes)
        y_pred = model.predict(X_sample)
        
        # Calculate accuracy
        acc = accuracy_score(y_sample, y_pred)
        results[f'Sampling{i+1}_Model{j+1}'] = acc

# Convert results to a DataFrame
import pandas as pd

results_df = pd.DataFrame(list(results.items()), columns=['Method', 'Accuracy'])
results_df[['Sampling', 'Model']] = results_df['Method'].str.split('_', expand=True)
results_df.drop(columns=['Method'], inplace=True)

best_results = results_df.groupby('Model')['Accuracy'].idxmax()
best_techniques = results_df.loc[best_results]

print("\nAccuracy for all sampling techniques and models:\n", results_df)
print("\nBest sampling technique for each model:\n", best_techniques)


# In[22]:





# In[ ]:




