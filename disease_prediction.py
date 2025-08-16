import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

# Step 1: Reading the dataset
data = pd.read_csv('improved_disease_dataset.csv')

# Encode disease labels into numbers
encoder = LabelEncoder()
data["disease"] = encoder.fit_transform(data["disease"])

# Features and target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Visualize class distribution before resampling
plt.figure(figsize=(18, 8))
sns.countplot(x=y)
plt.title("Disease Class Distribution Before Resampling")
plt.xticks(rotation=90)
plt.show()

# Step 2: Resampling the dataset using RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Step 3: Cross-validation with Stratified K-Fold
if 'gender' in X_resampled.columns:
    le = LabelEncoder()
    X_resampled['gender'] = le.fit_transform(X_resampled['gender'])

X_resampled = X_resampled.fillna(0)

if len(y_resampled.shape) > 1:
    y_resampled = y_resampled.values.ravel()

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

cv_scoring = 'accuracy'
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    try:
        scores = cross_val_score(
            model,
            X_resampled,
            y_resampled,
            cv=stratified_kfold,
            scoring=cv_scoring,
            n_jobs=-1,
            error_score='raise'
        )
        print("=" * 50)
        print(f"Model: {model_name}")
        print(f"Scores: {scores}")
        print(f"Mean Accuracy: {np.mean(scores):.4f}")
        print(f"Standard Deviation: {np.std(scores):.4f}")
        print("=" * 50)
    except Exception as e:
        print(f"Error with {model_name}: {e}")

# Step 4: Train the best model (Random Forest) on the entire resampled dataset
best_model = RandomForestClassifier()
best_model.fit(X_resampled, y_resampled)

# Step 5: Make predictions
sample_input = X_resampled.iloc[0].values.reshape(1, -1)
predicted_disease = encoder.inverse_transform(best_model.predict(sample_input))
print(f"Predicted Disease: {predicted_disease[0]}")
