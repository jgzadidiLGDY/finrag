import csv, sqlite3, os

# data file directories
DB_PATH = "data\\artifacts\\finrag.db"
CSV_PATH = "data\\raw\\financials.csv"
os.makedirs("data\\artifacts", exist_ok=True)

def seed():
    """
    The function populates a SQLite database with data from a CSV file
    """   
    # establishes a connection to a SQLite database.
    conn = sqlite3.connect(DB_PATH)
    # executes a SQL script to set up the database structure.
    conn.executescript(open("db/schema.sql").read())

    # reads the data from a CSV file
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    # begins a database transaction and performs a cleanup step.    
    with conn:
        conn.execute("DELETE FROM financials")  # idempotent seed
        #  inserts the data from the CSV into the database.
        #  it uses placeholders ? for each value to prevent SQL injection and properly handle data types
        conn.executemany(
            """INSERT INTO financials(ticker,fiscal_year,fiscal_quarter,revenue,eps)
               VALUES(?,?,?,?,?)""",
            [(r["ticker"].upper(), int(r["fiscal_year"]), r["fiscal_quarter"],
              float(r["revenue"]), float(r["eps"])) for r in rows]
        )
    # clean up and provide feedback.
    conn.close()
    print("Seeded:", len(rows), "rows")


if __name__ == "__main__": seed()
