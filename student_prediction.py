import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# -----------------------------
# Load Dataset
# -----------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(script_dir, "..", "DATA", "student_data.csv")

data = pd.read_csv(data_path)

print("\nDataset Preview:")
print(data.head())


# -----------------------------
# Data Visualization (EDA)
# -----------------------------

plt.figure(figsize=(6,4))
sns.countplot(data=data, x="Career")

plt.title("Career Distribution")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("../career_distribution.png")

plt.figure(figsize=(6,4))
sns.heatmap(data.drop("Career", axis=1).corr(), annot=True, cmap="coolwarm")

plt.title("Skill Correlation Heatmap")

plt.tight_layout()

plt.savefig("../correlation_heatmap.png")

# -----------------------------
# Data Preprocessing
# -----------------------------

X = data.drop("Career", axis=1)
y = data["Career"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)


# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# Model Training
# -----------------------------

model = RandomForestClassifier(n_estimators=200, random_state=42)

model.fit(X_train, y_train)


# -----------------------------
# Predictions
# -----------------------------

predictions = model.predict(X_test)


# -----------------------------
# Model Evaluation
# -----------------------------

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("../confusion_matrix.png")


# -----------------------------
# Feature Importance
# -----------------------------

importance = model.feature_importances_

importance_df = pd.Series(importance, index=X.columns)

plt.figure(figsize=(6,4))

importance_df.sort_values().plot(kind="barh")

plt.title("Feature Importance")
plt.xlabel("Importance Score")

plt.tight_layout()

plt.savefig("../feature_importance.png")