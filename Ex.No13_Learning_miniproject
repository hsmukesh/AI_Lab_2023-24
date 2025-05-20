# Ex.No: 13 Mini project - Gas Sensor Array Data Analysis for Low-Concentration Detection
### DATE:                                                                            
### REGISTER NUMBER : 212222060025
### AIM:
To analyze and process data from a gas sensor array to accurately detect and classify gases at low concentrations using data preprocessing and machine learning techniques.
###  Algorithm:
Step 1:Open Google Colab and create a new notebook.
Step 2:Import necessary libraries like NumPy, Pandas, Matplotlib, Scikit-learn, etc.
Step 3:Upload or mount the gas sensor dataset from local storage or Google Drive.
Step 4:Load the dataset into a Pandas DataFrame and display the first few rows.
Step 5:Check for and handle missing or duplicate data if present.
Step 6:Normalize or standardize the dataset using StandardScaler.
Step 7:Apply dimensionality reduction (e.g., PCA) to simplify feature space if needed.
Step 8:Split the dataset into training and testing sets.
Step 9:Train a classification model such as Random Forest or SVM on the training set.
Step 10:Evaluate model performance using accuracy, confusion matrix, and classification report.
Step 11:Visualize key results such as predictions, feature importance, or PCA scatter plot.

### Program:

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt



# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA to retain 95% variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA reduced to {X_pca.shape[1]} components")

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split for final evaluation after CV
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42)

# RandomForest with GridSearchCV for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")

# Best model evaluation on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Cross-validation score on the entire dataset for robustness
cv_scores = cross_val_score(best_model, X_pca, y_encoded, cv=5, scoring='accuracy')
print(f"\n5-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('/content/gsalc.csv')

# Extract Target
df['Gas_Type'] = df.iloc[:, 0]
df['Concentration'] = df.iloc[:, 1]

# Extract Numeric Sensor Data
#X = df.drop(columns=['Gas_Type', 'Concentration']).astype(float)
X = df.select_dtypes(include=[np.number])
y = df['Concentration']

# ----------------------------- #
# 1. Sample Distribution Plot
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Concentration', palette='viridis')
plt.title('Sample Distribution per Concentration Level')
plt.xlabel('Concentration Level')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------- #
# 2. Correlation Heatmap (First 20 sensors)
plt.figure(figsize=(12, 10))
subset_corr = X.iloc[:, :20].corr()
sns.heatmap(subset_corr, cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap (First 20 Sensor Readings)')
plt.show()

# ----------------------------- #
# 3. PCA Visualization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Concentration', palette='Set2')
plt.title('PCA Scatter Plot of Gas Sensor Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ----------------------------- #
# 4. Feature Importance (RandomForest)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y_encoded)

importances = rf.feature_importances_
indices = np.argsort(importances)[-20:]  # Top 20 important features

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align='center', color='teal')
plt.yticks(range(len(indices)), [f'Sensor {i}' for i in indices])
plt.xlabel('Relative Importance')
plt.title('Top 20 Feature Importances (RandomForest)')
plt.tight_layout()
plt.show()

# ----------------------------- #
# 5. PCA Explained Variance Plot
pca_full = PCA().fit(X_scaled)

plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.tight_layout()
plt.show()


### Output:


### Result:
Thus the system was trained successfully and the prediction was carried out.
