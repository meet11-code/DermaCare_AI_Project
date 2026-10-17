import sqlite3

conn = sqlite3.connect("users.db")
cursor = conn.cursor()

# Users table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fullname TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    mobile TEXT NOT NULL,
    password TEXT NOT NULL
)
""")

# Prediction history table
cursor.execute("""
CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    disease TEXT NOT NULL,
    confidence REAL,
    image TEXT,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    pdf TEXT
)
""")

conn.commit()
conn.close()

print("✅ Database created successfully!")
print("✅ users table created!")
print("✅ prediction_history table created!")