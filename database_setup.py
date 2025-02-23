import pandas as pd
import sqlite3

# Load CSV
csv_file = "Final_State_Specific_Farmer_Schemes.csv"  
df = pd.read_csv(csv_file)

print(df.shape)

# Connect to SQLite database
conn = sqlite3.connect("schemes.db")
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS schemes_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Scheme Name TEXT,
        Objective TEXT,
        Key Benefits TEXT,
        Eligibility TEXT,
        Implementing Agency TEXT,
        Source TEXT,
        State TEXT,
        All_Documents_Required TEXT
    )
''')

# Insert data into table
df.to_sql("schemes_data", conn, if_exists="replace", index=False)

# Commit and close
conn.commit()
conn.close()

print("Data stored in SQLite database 'schemes.db'")