# Quick Start Guide
QUICK START GUIDE - Python SimpleDB Database from University of Sydney

## Step 1: Verify Installation

Python 3.7+ is required. Check your installation:

```bash
python3 --version
```

## Step 2: Navigate to the Assignment code
This is the same directory, where this QUICK_START.md file resides.

```bash
cd "SimpleDB-Assignment"
```

## Step 3: Run the Demo
This is done directly from the root directory of the Assignment folder:

```bash
python3 -B -m simpledb.run.demo
```

You should see:

```bash
SimpleDB Query Engine
Type 'quit' to exit
SQL>
```

## Step 4: Try Sample Queries

### Create tables and insert data
This is done in the program code. The demo automatically sets up:
- `students` table with columns: name (STRING), age (INTEGER), class (STRING), male (BOOLEAN)
- `tutors` table with columns: id (STRING), tutor (STRING)

You can check the currently loaded schema using the 'schema' or 'tables' commands:
```bash
SimpleDB Query Engine
SQL> schema
```

### Sample Queries:

**1. Simple SELECT:**
```sql
SELECT name, age FROM Students;
```

**2. SELECT with projection:**
```sql
SELECT name, class FROM Students;
```

**3. JOIN query:**
```sql
SELECT name, tutor FROM Students JOIN Tutors ON class = id;
```

**4. Full SQL support including join, filtering, ordering and limit:**
```sql
SELECT name, age, class FROM Students JOIN Tutors ON class = id WHERE tutor=Scott ORDER BY name LIMIT 2;
```

## Running Tests

Directly from the root directory of the `SimpleDB-Assignment` folder:

```bash
python3 -B -m unittest discover -s tests -p "test_*.py" -v 2>&1
```

Or with pytest:

```bash
pytest tests/test_main/test_database_manager.py -v
```

## Module Overview

| Module | Purpose |
|--------|---------|
| `main/` | Global constants, database manager |
| `main/catalog` | Type system, schema |
| `heap/` | Tuple representation and heap file management |
| `disk/` | Disk manager and page I/O |
| `buffer/` | Buffer pool with MRU replacement |
| `access/` | Record access (read/write iterators) |
| `parser/` | SQL query parsing |
| `executor/` | Query engine with REPL |
| `executor/join/` | Join algorithm implementations |

## Creating Custom Queries

To use the database programmatically:

```python
from simpledb.main.database_manager import DatabaseManager
from simpledb.main.catalog.tuple_desc import TupleDesc

# Initialize database
dbms = DatabaseManager()

# Create a schema
schema = TupleDesc()
schema.add_string("name").add_integer("age")
dbms.get_catalog().add_schema(schema, "my_table")

# Get the table
table = dbms.get_heap_file("my_table")

# Insert rows
with table.inserter() as inserter:
    inserter.insert(["Alice", 30])
    inserter.insert(["Bob", 25])

# Read data
for tuple_obj in table.iterator():
    print(tuple_obj)

#SQL usage
QueryEngine(dbms).run()
```

## Common Issues

### Issue: Module not found
**Solution**: Make sure you're running from the correct directory
and that you run the code as module using the '-m' option. 
```bash
cd "/path/to/SimpleDB"
python3 -B -m simpledb.run.demo
```

### Issue: Query syntax error
**Solution**: Ensure your query follows this format:
```sql
SELECT col1, col2 FROM table [JOIN table2 ON col1 = col2];
```
where col1 must refer to an attribute in table, and col2 to table2.

### Issue: Table/Column not found
**Solution**: 
1. Verify the schema was added: `dbms.get_catalog().read_schema("table_name")`
2. Check column names match exactly (case-sensitive)

## Performance Tips

1. **Buffer Pool**: Configured with 32 frames - adjust in `database_constants.py`
  or using the runtime parameters of demo.py

simpledb.run.demo can be started with two optional parameters:

```bash
python -B -m simpledb.run.demo -h   
usage: demo.py [-h] [-d FILNAME] [-b SIZE]

SimpleDB demo

options:
  -h, --help            show this help message and exit
  -d, --dbfile FILNAME  name of database file
  -b, --buffer SIZE     number of buffer frames
```

## Project Structure

```
Join-Algorithms/
├── simpledb/            # Main implementation
├── tests/               # Unit tests
├── README.md            # Full documentation
├── QUICK_START.md       # This file
```
---

**Happy database exploring!**
