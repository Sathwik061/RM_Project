import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    recall_score
)
df=pd.read_csv(r"D:\RM\solar+flare\data-processed.csv")
# X = input features
# y = target (0 = low risk, 1 = high risk)

# High-risk if M or X flare count > 0
df['target'] = ((df['class_mc_flares'] > 0) | (df['class_x_flares'] > 0)).astype(int)


#  CORRECT FEATURE–TARGET SEPARATION (NO DATA LEAKAGE)
X = df.drop(columns=[
    'target',
    'class_code',
    'class_mc_flares',
    'class_x_flares'
])

y = df['target']


# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_cols)
#APPLY LABEL ENCODING (FIX)
from sklearn.preprocessing import LabelEncoder

categorical_cols = X.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_cols)

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y   # IMPORTANT for imbalanced data
)

#Baseline model
baseline_model = DecisionTreeClassifier(
    random_state=42
)

baseline_model.fit(X_train, y_train)

y_pred_baseline = baseline_model.predict(X_test)

#COST-SENSITIVE DECISION TREE (MAIN MODEL)

class_weights = {
    0: 1,   # Low-risk
    1: 50    # High-risk (penalize misclassification more)
}
cost_sensitive_model = DecisionTreeClassifier(
    class_weight=class_weights,
    random_state=42
)

cost_sensitive_model.fit(X_train, y_train)

y_pred_cost = cost_sensitive_model.predict(X_test)

def evaluate_model(y_true, y_pred, model_name):
    print(f"\n===== {model_name} =====")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Recall (High-Risk):", recall_score(y_true, y_pred, pos_label=1))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

#Evaluating the both the models
evaluate_model(y_test, y_pred_baseline, "Baseline Decision Tree")

evaluate_model(y_test, y_pred_cost, "Cost-Sensitive Decision Tree")
