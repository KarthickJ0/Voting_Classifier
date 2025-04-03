import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
data = pd.read_csv(r"C:\Users\Vetri\OneDrive\Desktop\karthick J\Project\path_to_cleaned_data.csv")  # Replace with your actual dataset file path
print("Columns in the dataset:", data.columns)

# Split the data into features (X) and target (y)
X = data.drop(columns=['survival_status'])  # Replace with the correct target column
y = data['survival_status']  # Replace with the correct target column

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define individual models
logistic_model = LogisticRegression(C=0.1, solver='liblinear', max_iter=500, random_state=42)
random_forest_model = RandomForestClassifier(n_estimators=50, max_depth=10, criterion='gini', random_state=42)
svm_model = SVC(C=10, kernel='rbf', probability=True, random_state=42)

# Train individual models
print("Training Logistic Regression...")
logistic_model.fit(X_train, y_train)

print("Training Random Forest...")
random_forest_model.fit(X_train, y_train)

print("Training SVM...")
svm_model.fit(X_train, y_train)

# Hard Voting Classifier
print("\nTraining Hard Voting Classifier...")
voting_clf_hard = VotingClassifier(
    estimators=[
        ('lr', logistic_model),
        ('rf', random_forest_model),
        ('svm', svm_model)
    ],
    voting='hard'  # Majority rule
)
voting_clf_hard.fit(X_train, y_train)

# Soft Voting Classifier
print("\nTraining Soft Voting Classifier...")
voting_clf_soft = VotingClassifier(
    estimators=[
        ('lr', logistic_model),
        ('rf', random_forest_model),
        ('svm', svm_model)
    ],
    voting='soft'  # Weighted probabilities
)
voting_clf_soft.fit(X_train, y_train)

# Stacking Classifier
print("\nTraining Stacking Classifier...")
stacking_clf = StackingClassifier(
    estimators=[
        ('lr', logistic_model),
        ('rf', random_forest_model),
        ('svm', svm_model)
    ],
    final_estimator=LogisticRegression(max_iter=500, random_state=42),
    cv=5
)
stacking_clf.fit(X_train, y_train)

# Evaluate all classifiers
print("\nEvaluating Models:")
models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model,
    "SVM": svm_model,
    "Hard Voting": voting_clf_hard,
    "Soft Voting": voting_clf_soft,
    "Stacking": stacking_clf
}

for model_name, model in models.items():
    print(f"\nModel: {model_name}")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Save the ensemble models
print("\nSaving Models...")
joblib.dump(voting_clf_hard, 'voting_classifier_hard.pkl')
joblib.dump(voting_clf_soft, 'voting_classifier_soft.pkl')
joblib.dump(stacking_clf, 'stacking_classifier.pkl')
print("Models saved successfully!")
