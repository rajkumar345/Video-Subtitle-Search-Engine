import sqlite3
import pandas as pd

def read_database(db_path):
    # Step 1 - Connect to SQLite Database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Step 2 - Fetch Table Names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print("Tables in Database:", tables)
    
    # Step 3 - Read Table 'zipfiles'
    df = pd.read_sql_query("SELECT * FROM zipfiles", conn)
    
    # Step 4 - Save to CSV
    df.to_csv("subtitles_raw.csv", index=False)
    
    # Close connection
    conn.close()
    return df

if __name__ == "__main__":
    df = read_database("eng_subtitles_database.db")
    print(df.head())
