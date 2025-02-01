import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the CSV data into a pandas DataFrame
df = pd.read_csv("heart.csv")

# Step 1: Remove duplicates
df = df.drop_duplicates()

# Step 2: Handle missing values (e.g., fill NaN with the mean for numerical columns)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Cholesterol'].fillna(df['Cholesterol'].mean(), inplace=True)

# Step 3: Convert categorical variables to correct data types if needed
df['Sex'] = df['Sex'].astype('category')  # Ensures 'Sex' column is categorical
df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})  # Map 'Y'/'N' to 1/0

# Step 4: Handle any outliers or incorrect values
df['RestingBP'] = df['RestingBP'].apply(lambda x: 200 if x > 200 else x)

# Step 5: Normalize/standardize values if required (for example, scaling features like 'Age' or 'Cholesterol')
scaler = StandardScaler()
df[['Age', 'Cholesterol']] = scaler.fit_transform(df[['Age', 'Cholesterol']])

# Now the DataFrame is cleaned and can be inserted into the database
print(df.head())  # Print cleaned data
