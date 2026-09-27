import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv("student_placement_dataset.csv")

# Clean dataset
df.drop("Student_ID", axis=1, inplace=True)

# Encode Branch and Gender as single columns
le_branch = LabelEncoder()
df['Branch'] = le_branch.fit_transform(df['Branch'])

le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

# Encode target variable
df['Placed'] = df['Placed'].map({'Yes':1, 'No':0})

# Save encoders for use in app.py
joblib.dump(le_branch, "branch_encoder.pkl")
joblib.dump(le_gender, "gender_encoder.pkl")

# Features and targets
X = df.drop(["Placed","Salary"], axis=1)
y_class = df["Placed"]
y_reg = df["Salary"]

# Save expected feature order
expected_order = list(X.columns)
joblib.dump(expected_order, "expected_order.pkl")

# -------------------------------
# Classification model with SMOTE
# -------------------------------
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_class)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred_class, zero_division=0))
joblib.dump(clf, "placement_model.pkl")

# -------------------------------
# Regression model (unchanged)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
reg = RandomForestRegressor(random_state=42)
reg.fit(X_train, y_train)
y_pred_reg = reg.predict(X_test)

print("\n=== Regression Results ===")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_reg))
joblib.dump(reg, "salary_model.pkl")

print("\n✅ Models trained and saved as placement_model.pkl, salary_model.pkl, branch_encoder.pkl, and gender_encoder.pkl")
