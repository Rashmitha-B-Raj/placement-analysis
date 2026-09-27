import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv("student_placement_dataset.csv")

# Step 2: Drop Student_ID (not useful for prediction)
df.drop("Student_ID", axis=1, inplace=True)

# Step 3: Encode categorical variables (Branch, Gender)
df = pd.get_dummies(df, columns=['Branch','Gender'], drop_first=True)

# Step 4: Encode target variable (Placed: Yes=1, No=0)
df['Placed'] = df['Placed'].map({'Yes':1, 'No':0})

# Step 5: Preview the cleaned dataset
print(df.head())
print(df.info())
