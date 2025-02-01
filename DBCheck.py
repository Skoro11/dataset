import os
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
