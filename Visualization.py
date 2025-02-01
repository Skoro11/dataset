import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sqlalchemy import create_engine
import psycopg2
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load environment variables from a .env file
load_dotenv()

# Step 2: Retrieve database credentials from the environment variables
db_username = os.getenv('DB_USERNAME')
db_password = os.getenv('DB_PASSWORD')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
db_host = os.getenv('DB_HOST')

# Step 3: Establish a connection to PostgreSQL using psycopg2
try:
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_username,
        password=db_password,
        host=db_host,
        port=db_port
    )
    print("Database connection successful!")
except Exception as e:
    print("Error while connecting to PostgreSQL", e)

# Step 4: Load the CSV data into a pandas DataFrame
df = pd.read_csv("heart.csv")

# Step 5: Remove duplicates
df = df.drop_duplicates()

# Step 6: Handle missing values (filling with the mean for numerical columns)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Cholesterol'] = df['Cholesterol'].fillna(df['Cholesterol'].mean())
df['RestingBP'] = df['RestingBP'].fillna(df['RestingBP'].mean())

# Step 7: Convert categorical variables to correct data types if needed
df['Sex'] = df['Sex'].astype('category')  # Convert 'Sex' column to categorical type
df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})  # Map 'Y'/'N' to 1/0
df['RestingECG'] = df['RestingECG'].astype('category')  # Convert 'RestingECG' to categorical
df['ST_Slope'] = df['ST_Slope'].astype('category')  # Convert 'ST_Slope' to categorical
df['ChestPainType'] = df['ChestPainType'].astype('category')  # Convert 'ChestPainType' to categorical

# Step 8: Handle outliers (e.g., cap 'RestingBP' values above 200)
df['RestingBP'] = df['RestingBP'].apply(lambda x: 200 if x > 200 else x)

# Step 9: Standardize numerical features (for example, scaling 'Age' and 'Cholesterol')
scaler = StandardScaler()
df[['Age', 'Cholesterol']] = scaler.fit_transform(df[['Age', 'Cholesterol']])

# Step 10: Convert categorical features to numeric for correlation calculation
label_encoder = LabelEncoder()

# Convert categorical variables to numeric using LabelEncoder
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['RestingECG'] = label_encoder.fit_transform(df['RestingECG'])
df['ST_Slope'] = label_encoder.fit_transform(df['ST_Slope'])
df['ChestPainType'] = label_encoder.fit_transform(df['ChestPainType'])

# Step 11: Exclude the HeartDisease column from the correlation matrix calculation
df_features = df.drop(columns=['HeartDisease'])

# Step 12: Calculate the correlation matrix (now that all data is numeric)
correlation_matrix = df_features.corr()

# Step 13: Extract the correlation with HeartDisease (target column)
correlation_with_target = df.corr()['HeartDisease'].sort_values(ascending=False)

# Step 14: Define explanations for each variable
explanations = {
    'Age': "Age of the patient (years). A positive correlation means older individuals are more likely to develop heart disease.",
    'Sex': "Gender of the patient (Male = 1, Female = 0). A positive correlation means men are more likely to develop heart disease.",
    'ChestPainType': "Type of chest pain experienced by the patient. A positive correlation suggests that certain types of chest pain, especially angina, are linked to heart disease.",
    'RestingBP': "Resting blood pressure (mm Hg). A positive correlation means that higher blood pressure is associated with greater heart disease risk.",
    'Cholesterol': "Serum cholesterol levels (mg/dl). A positive correlation means higher cholesterol levels are linked to heart disease.",
    'FastingBS': "Fasting blood sugar levels (1 if > 120 mg/dl, 0 otherwise). A positive correlation suggests that higher fasting blood sugar increases heart disease risk.",
    'RestingECG': "ECG results at rest. A positive correlation means abnormal ECG readings (like ST-T wave abnormalities) are associated with heart disease.",
    'MaxHR': "Maximum heart rate achieved during exercise. A negative correlation suggests that higher heart rate during exercise is associated with lower risk of heart disease.",
    'ExerciseAngina': "Whether the patient experiences chest pain during exercise (Yes = 1, No = 0). A positive correlation means that chest pain during exercise is linked to heart disease.",
    'Oldpeak': "Depression in ST segment during exercise (measured in depression). A positive correlation means greater ST depression is linked to heart disease.",
    'ST_Slope': "The slope of the peak exercise ST segment. A positive correlation means that certain ST slopes are linked to heart disease."
}

# Step 15: Display the explanations and correlations
print("Correlations with Heart Disease (Human-readable format):\n")
for feature, correlation in correlation_with_target.items():
    print(f"\n--- {feature} ---")
    print(explanations.get(feature, "No explanation available."))
    
    if correlation > 0.5:
        print(f"Correlation with Heart Disease: +{correlation:.2f} (strong positive correlation, meaning higher values are strongly associated with heart disease)")
    elif correlation > 0.2:
        print(f"Correlation with Heart Disease: +{correlation:.2f} (moderate positive correlation, meaning higher values are somewhat associated with heart disease)")
    elif correlation < -0.2:
        print(f"Correlation with Heart Disease: -{correlation:.2f} (moderate negative correlation, meaning higher values are somewhat associated with lower heart disease risk)")
    elif correlation < -0.5:
        print(f"Correlation with Heart Disease: -{correlation:.2f} (strong negative correlation, meaning higher values are strongly associated with lower heart disease risk)")
    else:
        print(f"Correlation with Heart Disease: {correlation:.2f} (weak or negligible correlation, meaning no strong relationship with heart disease)")

# Example for Cholesterol based on a negative correlation value:
if 'Cholesterol' in correlation_with_target:
    cholesterol_corr = correlation_with_target['Cholesterol']
    if cholesterol_corr < 0:
        print(f"Cholesterol's correlation with Heart Disease: {cholesterol_corr:.2f} (weak negative correlation, meaning that higher cholesterol levels might be weakly associated with lower heart disease risk in this dataset. This could be due to other factors or data anomalies.)")
    else:
        print(f"Cholesterol's correlation with Heart Disease: {cholesterol_corr:.2f} (positive correlation, meaning higher cholesterol levels are associated with greater heart disease risk)")

# Step 16: Plot the correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()

# Step 17: Plot the correlation with HeartDisease as a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=correlation_with_target.index, y=correlation_with_target.values)
plt.title('Correlation of Features with Heart Disease')
plt.xticks(rotation=45)
plt.show()
