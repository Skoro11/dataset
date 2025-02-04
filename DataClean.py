import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the CSV data into a pandas DataFrame
df = pd.read_csv("heart.csv")

# Step 2: Remove duplicates
df = df.drop_duplicates()

# Step 3: Handle missing values (filling with the mean for numerical columns)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Cholesterol'] = df['Cholesterol'].fillna(df['Cholesterol'].mean())
df['RestingBP'] = df['RestingBP'].fillna(df['RestingBP'].mean())
df['MaxHR'] = df['MaxHR'].fillna(df['MaxHR'].mean())  # Added missing handling for 'MaxHR'

# Step 4: Convert categorical variables to numeric using Label Encoding
label_encoder = LabelEncoder()

df['Sex'] = label_encoder.fit_transform(df['Sex'])  # Convert 'Sex' column to numeric (0, 1)
df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})  # Map 'Y'/'N' to 1/0
df['RestingECG'] = label_encoder.fit_transform(df['RestingECG'])  # Convert 'RestingECG' to numeric
df['ST_Slope'] = label_encoder.fit_transform(df['ST_Slope'])  # Convert 'ST_Slope' to numeric

# Step 5: Handle outliers (e.g., cap 'RestingBP' values above 200)
df['RestingBP'] = df['RestingBP'].apply(lambda x: 200 if x > 200 else x)

# Step 6: Calculate correlation matrix
numerical_df = df.select_dtypes(include=['float64', 'int64'])  # Select only numerical columns
correlation_matrix = numerical_df.corr()

# Visualize correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Step 7: Visualize distributions (Histograms)
plt.figure(figsize=(14, 8))
sns.histplot(df['Age'], kde=True, color='blue', label='Age')
sns.histplot(df['Cholesterol'], kde=True, color='green', label='Cholesterol')
sns.histplot(df['RestingBP'], kde=True, color='red', label='RestingBP')
sns.histplot(df['MaxHR'], kde=True, color='purple', label='MaxHR')
plt.legend()
plt.title('Distribution of Numerical Features')
plt.show()

# Step 8: Box Plots to visualize distributions and outliers
plt.figure(figsize=(14, 8))
sns.boxplot(data=df[['Age', 'Cholesterol', 'RestingBP', 'MaxHR']])
plt.title('Box Plots for Age, Cholesterol, RestingBP, and MaxHR')
plt.show()

# Step 9: Pairplot (Scatterplot Matrix) for numerical features
sns.pairplot(df[['Age', 'Cholesterol', 'RestingBP', 'MaxHR', 'HeartDisease']], hue='HeartDisease', palette='coolwarm')
plt.title('Pairplot of Numerical Features Colored by Heart Disease')
plt.show()

# Step 10: Bar Plot for categorical variables (Gender, Exercise Angina)
plt.figure(figsize=(14, 8))
sns.countplot(x='Sex', hue='HeartDisease', data=df, palette='coolwarm')
plt.title('Bar Plot of Sex vs Heart Disease')
plt.show()

# Step 11: Violin Plot for numerical features segmented by Heart Disease
plt.figure(figsize=(14, 8))
sns.violinplot(x='HeartDisease', y='Age', data=df, palette='coolwarm')
plt.title('Violin Plot of Age vs Heart Disease')
plt.show()

plt.figure(figsize=(14, 8))
sns.violinplot(x='HeartDisease', y='Cholesterol', data=df, palette='coolwarm')
plt.title('Violin Plot of Cholesterol vs Heart Disease')
plt.show()

# Step 12: Count Plot for categorical features like Exercise Angina and RestingECG
plt.figure(figsize=(14, 8))
sns.countplot(x='ExerciseAngina', hue='HeartDisease', data=df, palette='coolwarm')
plt.title('Exercise Angina vs Heart Disease')
plt.show()

plt.figure(figsize=(14, 8))
sns.countplot(x='RestingECG', hue='HeartDisease', data=df, palette='coolwarm')
plt.title('RestingECG vs Heart Disease')
plt.show()

# Step 13: Count Plot for ST_Slope vs Heart Disease
plt.figure(figsize=(14, 8))
sns.countplot(x='ST_Slope', hue='HeartDisease', data=df, palette='coolwarm')
plt.title('ST Slope vs Heart Disease')
plt.show()

# Optional: You can add more specific charts for any additional features as needed.
