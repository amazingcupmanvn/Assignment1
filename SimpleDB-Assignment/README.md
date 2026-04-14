# SimpleDB - Python Database Engine

This is the source code of a simple Python database engine for teaching DATA3404 
at the University of Sydney.

## Project Structure

```
simpledb/
├── main/                   # Global constants and configuration
    └── catalog/            # Catalog and Type representations
├── heap/                   # Heap file management and tuple representation
├── disk/                   # Disk management and page I/O
├── buffer/                 # Buffer pool management
├── access/                 # Record access (read/write iterators)
├── parser/                 # Query parsing
├── executor/               # Query execution engine
    ├── projection/         # Column projection
    ├── join/               # Join algorithm implementations
    ├── filter/             # Filter operator implementing WHERE clause
    ├── ordering/           # OrderBy operator implementing ORDER BY clause
    └── limit/              # Limit operator imlementing LIMIT clause
└── run/                    # Demo application

tests/                      # Unit tests
├── test_main/              # unit tests of database manager
├── test_buffer/            # unit tests for the buffer manager
├── test_parser/            # unit tests for query parsing
└── test_executor/          # unit tests for query executor
```

## Key Features

- **Query Execution**:
  - Pipelined query execution
  - Separate SQL query parser, planner and executor stages
  - NO query optimiser; only bare-bone parser and planner
  
- **Join Algorithms**:
  - Nested-Loops Join
    Current restriction: Join between just two relations

- **Database Management**:
  - Buffer pool with Random replacement policy
  - Row-store with slotted page architecture
  - Disk manager with file I/O, storing all tables within a single database file
  - Catalog for schema management (only table names made persistent, other schema in code)

- **Type System**:
  - Support for STRING, INTEGER, DOUBLE, and BOOLEAN types
  - Type-safe tuple operations
  - Schema validation

## Installation

No external dependencies are required - uses only Python standard library.

```bash
cd SimpleDB-Assignment
```

## Running the Demo
This is done directly from the root directory of the cloned assignment repo:

```bash
python3 -B -m simpledb.run.demo
```

This starts an interactive query engine where you can:
- inspect the schema (schema command)
- Execute SELECT and JOIN queries


Example queries:
```sql
SELECT name, age   FROM Students;
SELECT name, tutor FROM Students JOIN Tutors ON class = id;
```

## Running Unit Tests

```bash
python3 -B -m unittest discover -s tests -p "test_*.py" -v 2>&1
```

## Implementation Notes

### Key Classes

- `Tuple`: Represents a row in the database
- `TupleDesc`: Schema definition
- `Page/DataPage`: Fixed-size disk pages (1KB)
- `DiskManager`: Handles disk I/O
- `BufferManager`: Manages buffer pool
- `HeapFile`: Collection of data pages
- `AccessIterator`: Abstract iterator for table access
- Join algorithms: NestedLoopJoin

## Architecture

The system follows a layered architecture:

```
Query Engine
    ↓
Query Parser
    ↓  
Query Planner
    ↓  
Join Operators + Projection
    ↓
Access Layer (Iterators)
    ↓
Buffer Manager
    ↓
Disk Manager
```

## Performance Considerations

- Buffer pool currently uses a random replacement policy
- Page size: 1024 bytes
- Maximum buffer frames: 32

## Acknowledgements
SimpleDB is based on the PASTA JavaDB from INFO3404/DATA3404 which was mainly
written by Scott Sidwell, Chris Natoli, and Bryn Jeffries.

## References

- Original Java Project: PASTA/JavaDB (University of Sydney, DATA3404/INFO3404)
- O'Reilly "Database Internals" for disk management concepts
- Ramakrishnan/Gehrke: "Database Management Systems"

---

**Note**: This is an educational implementation whose complexity is purposely simplified. 
For production use, consider established databases like PostgreSQL, MySQL, or SQLite.
