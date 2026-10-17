import sqlite3

# ============================================================
# DATABASE CONNECTION
# ============================================================

conn = sqlite3.connect("users.db")

cursor = conn.cursor()


# ============================================================
# USERS TABLE
# ============================================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    fullname TEXT NOT NULL,

    email TEXT UNIQUE NOT NULL,

    mobile TEXT NOT NULL,

    password TEXT NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

)
""")


# ============================================================
# PREDICTION HISTORY TABLE
# ============================================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS prediction_history(

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    username TEXT NOT NULL,

    disease TEXT NOT NULL,

    confidence REAL NOT NULL,

    image TEXT,

    pdf TEXT,

    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP

)
""")

# ============================================================
# ADD CATEGORY COLUMN TO PREDICTION HISTORY
# ============================================================

try:

    cursor.execute("""
        ALTER TABLE prediction_history
        ADD COLUMN category TEXT DEFAULT 'human'
    """)

    print("category column added successfully.")

except sqlite3.OperationalError as e:

    if "duplicate column name" in str(e).lower():

        print("category column already exists.")

    else:

        print(
            "Category column migration error:",
            e
        )


# ============================================================
# SAVE CHANGES
# ============================================================

conn.commit()

conn.close()


print("=" * 60)
print("DATABASE INITIALIZATION SUCCESSFUL")
print("=" * 60)
print("users table              : READY")
print("prediction_history table : READY")
print("Database                 : users.db")
print("=" * 60)