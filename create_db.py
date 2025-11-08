# a script to creat finRAG db 
# imports the built-in Python module for interacting with SQLite databases
import sqlite3

#  establishes a connection to a specific SQLite database file
conn = sqlite3.connect("data/artifacts/finrag.db")

# The executescript() method can execute multiple SQL statements at once. 
# It is perfectly suited for initializing a database from a script file, 
# as it handles the separate statements and their dependencies in order.
with open("db/schema.sql") as f:
    conn.executescript(f.read())

#  queries the database to confirm that the table was created successfully
print("tables:", list(conn.execute("SELECT name FROM sqlite_master WHERE type='table'")))
