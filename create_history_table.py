import sqlite3

conn = sqlite3.connect("users.db")

cursor = conn.cursor()

cursor.execute("""

CREATE TABLE IF NOT EXISTS prediction_history(

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    username TEXT,

    disease TEXT,

    confidence REAL,

    image TEXT,

    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP

)

""")

conn.commit()

conn.close()

print("Prediction History Table Created Successfully ✅")