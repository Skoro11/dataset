Overview
This project consists of three main Python scripts used for data cleaning, database verification, and data visualization. These scripts work together to analyze heart disease data from a Kaggle dataset.

DataClean.py: This script is responsible for cleaning the data extracted from a CSV file downloaded from Kaggle. It handles tasks like removing duplicates, handling missing values, and formatting the data for further analysis.

DBCheck.py: This script checks if the connection is successfully established between the program and the database where the cleaned data is stored. It ensures that the data is accessible for further querying or manipulation.

Visualization.py: This script generates visualizations, including graphs and charts, that show the correlation between heart disease and various symptoms. It helps to better understand how different symptoms relate to the likelihood of heart disease.

Requirements
Python 3.x
Required Python libraries (listed in requirements.txt)
A database (e.g., MySQL, SQLite) for storing cleaned data
