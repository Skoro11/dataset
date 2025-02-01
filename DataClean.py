import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sqlalchemy import create_engine
import psycopg2
from dotenv import load_dotenv

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

# Step 8: Handle outliers (e.g., cap 'RestingBP' values above 200)
df['RestingBP'] = df['RestingBP'].apply(lambda x: 200 if x > 200 else x)

# Step 9: Normalize the 'Cholesterol' column (to ensure higher cholesterol gives higher positive correlation)
scaler = MinMaxScaler()
df[['Age']] = scaler.fit_transform(df[['Age']])  # Normalize 'Age'
df[['Cholesterol']] = scaler.fit_transform(df[['Cholesterol']])  # Normalize 'Cholesterol'

# Step 10: Print the cleaned data to check the result
print(df.head())

# Step 11: Create a connection to PostgreSQL using SQLAlchemy
DATABASE_URL = f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"

# Step 12: Create SQLAlchemy engine to interact with PostgreSQL
engine = create_engine(DATABASE_URL)

# Step 13: Write the cleaned DataFrame to the PostgreSQL database
try:
    df.to_sql('heart_disease', con=engine, if_exists='replace', index=False)
    print("Data has been cleaned and imported into PostgreSQL successfully!")
except Exception as e:
    print("Error while inserting data into PostgreSQL", e)
