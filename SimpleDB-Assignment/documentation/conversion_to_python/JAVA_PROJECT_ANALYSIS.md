# Join Algorithms Project - Comprehensive Code Analysis

## Project Overview
This is a **database system implementation for studying join algorithms** written in Java. It provides a complete database engine with:
- Buffer management and disk I/O handling
- Heap file storage structures
- Three join algorithm implementations (Nested Loop, Block Nested Loop, Sort-Merge)
- Query execution engine
- Data persistence and serialization

---

## Build Configuration (build.xml)
**Build System:** Apache Ant
**Key Targets:**
- `TestEasy`: Creates test zip for Nested Loop Join
- `TestMedium`: Creates test zip for Block Nested Loop Join
- `TestHard`: Creates test zip for Sort-Merge Join
- `SolutionZip`: Creates complete solution archive

---

## Package Structure & Class Hierarchy

### 1. **global/** - Global Configuration & Management
Manages database-wide configuration and component initialization.

#### Classes:
**Type (Enum)**
- Purpose: Defines supported database data types
- Values: STRING, DOUBLE, INTEGER, BOOLEAN
- Methods:
  - `getLen()`: Returns byte size for each type
  - `parseType(String)`: Converts string to typed object
  - `getTypeClass()`: Returns Java Class of the type
- Fields: `typeClass` (Class<?>) - Java type mapping

**DatabaseConstants (Interface)**
- Static configuration constants:
  - `PAGE_SIZE = 1024` bytes
  - `MAX_BUFFER_FRAMES = 32`
  - `MAX_TABLE_NAME_LENGTH = 30`
  - `MAX_STRING_LENGTH = 15`
  - `FIRST_PAGE_ID = 0`
  - `INVALID_PAGE_ID = -1`

**Catalog**
- Purpose: Catalog/schema registry for the database
- Fields: `schemas: HashMap<String, TupleDesc>`
- Methods:
  - `addSchema(TupleDesc, String)`: Register a new schema
  - `readSchema(String)`: Retrieve a schema by name
  - `findNameOfSchema(TupleDesc)`: Reverse lookup

**DatabaseManager**
- Purpose: Central component manager and initialization
- Fields:
  - `isInitialised: boolean`
  - `dm: DiskManager`
  - `bm: BufferManager`
  - `catalog: Catalog`
- Methods:
  - `__init__()`: Initialize all components
  - `resetComponents()`: Reset to defaults
  - `getCatalog()`, `getDiskManager()`, `getBufferManager()`: Accessors
  - `getHeapFile(String)`: Load a file by relation name
  - `getTempHeapFile(TupleDesc)`: Create temporary file
  - `close()`: Clean up resources
- Exceptions: `ComponentsNotInitialisedError`

---

### 2. **heap/** - Tuple & Schema (In-Memory Data Structures)
Implements row-based data representation.

#### Classes:

**Tuple**
- Purpose: Represents a single row in the database
- Fields:
  - `schema: TupleDesc`
  - `pageId: PageId` - Which page contains this tuple
  - `slotId: int` - Position within the page (-1 if not set)
  - `row: Object[]` - The actual data values
- Key Methods:
  - `__init__(TupleDesc)` / `__init__(TupleDesc, Object[])`
  - `getColumn(int/String)`: Retrieve column value
  - `setColumn(int/String, Object)`: Set column with type checking
  - `getSchema()`, `getPageId()`, `getSlotId()`: Accessors
  - `copyValues(Tuple)`: Copy matching columns from another tuple
  - `rowEquals(Object...)`: Compare row values
  - `equals(Tuple)`: Tuple equality
  - `toString()`, `toRow()`: String representation

**TupleDesc** (Schema/Metadata)
- Purpose: Describes the schema of tuples
- Inner Class: `TupleDescItem` - Type-name pairs
- Fields: `columns: List<TupleDescItem>`
- Key Methods:
  - `addString/Integer/Double/Boolean(String)`: Build schema fluently
  - `getNumFields()`: Number of columns
  - `getFieldName(int)`, `getFieldType(int/String)`: Metadata access
  - `getIndexFromName(String)`: Find column position
  - `getColumnNames()`: List all column names
  - `getMaxTupleLength()`: Total byte size
  - `hasField(String)`: Check column exists
  - **Static** `join(TupleDesc, TupleDesc)`: Merge two schemas
  - **Static** `project(String...)`: Create reduced schema
- Exceptions: `NoSuchElementException`

**HeapFile**
- Purpose: Collection of unordered pages containing tuples
- Fields:
  - `schema: TupleDesc`
  - `firstPageId: PageId`
  - `bufferManager: BufferManager`
  - `relationName: String`
- Methods:
  - `__init__(TupleDesc, String, BufferManager)`: Named file
  - `__init__(TupleDesc, BufferManager)`: Temporary file (auto-named)
  - `iterator()` → AccessIterator: Create read iterator
  - `inserter()` → AccessInserter: Create write iterator
  - `getSchema()`, `printStats()`
- Creates pages in header during construction

**HeapPage**
- Purpose: Single page of heap file tuples
- Extends: `DataPage`
- Fields: `schema: TupleDesc`
- Methods:
  - `__init__(Page, TupleDesc)`
  - `iterator()`: Create iterator over page tuples

---

### 3. **disk/** - Physical Storage & Paging
Manages page-level I/O and storage structures.

#### Classes:

**Page**
- Purpose: Raw byte array container for 1KB pages
- Fields: `data: byte[PAGE_SIZE]`
- Methods:
  - `__init__()` / `__init__(byte[])`
  - `getData()`, `setData(byte[])`
  - `copy(Page)`: Copy entire page content
  - **Type accessors:**
    - `getIntegerValue(offset)`, `setIntegerValue(value, offset)` (4 bytes)
    - `getDoubleValue(offset)`, `setDoubleValue(value, offset)` (8 bytes)
    - `getBooleanValue(offset)`, `setBooleanValue(value, offset)` (1 byte)
    - `getStringValue(offset)`, `setStringValue(value, offset)` (2 + length bytes)

**PageId**
- Purpose: Identifier for disk pages
- Fields: `pageId: int`
- Methods:
  - `__init__()`, `__init__(int)`
  - `isValid()`: Check if ID != -1
  - `get()`, `set(int)`
  - `equals(Object)`, `hashCode()`: HashMap compatible
  - `toString()`

**DataPage (Abstract)** extends Page
- Purpose: Base for file pages with header and record slots
- Static Fields:
  - `PREV_PAGE_POS = 0`, `NEXT_PAGE_POS = 4`, `RECORD_COUNT_POS = 8`
  - `RELATION_NAME_POS = 12`, `RECORD_START_POS = 12 + 2 + 30`
- Methods:
  - `initialise(String)`: Set up new page with relation name
  - `getPreviousPageId()`, `setPreviousPageId(PageId)`: Linked list support
  - `getNextPageId()`, `setNextPageId(PageId)`
  - `getRecordCount()`, `setRecordCount(int)`: Number of tuples
  - `getRelationName()`, `setRelationName(String)`
  - `insertRecord(Tuple)` / `insertRecord(int, Tuple)`: Add tuple
  - `getRecord(int, Tuple)`: Retrieve tuple at slot
  - **Static** `getMaxRecordsOnPage(Tuple/TupleDesc)`: Capacity calculation
  - **Private** `read/write(Tuple, offset)`: Serialization

**HeaderPage** extends Page
- Purpose: Catalog page linking relation names to first page IDs
- Static Fields:
  - `NEXT_PAGE_INDEX = 0`, `NUM_POINTERS_INDEX = 4`
  - `POINTER_ENTRY_SIZE = 4 + 2 + 30` (pageId + name length + name)
- Methods:
  - `__init__()`, `__init__(Page)`
  - `initialise()`: Set up default header
  - `getNextPage()`, `setNextPage(PageId)`: Header chain
  - `getNumPointers()`, `setNumPointers(int)`: Max entries
  - `getFileEntry(int, PageId)`: Get relation entry
  - `setFileEntry(int, PageId, String)`: Set relation entry
  - **Static** `getFileEntry(BufferManager, String)` → PageId: Look up file
  - **Static** `setFileEntry(BufferManager, String, PageId)`: Register file
- Exceptions: AssertionError

**DiskManager**
- Purpose: File-based persistent storage I/O
- Fields:
  - `dbFile: RandomAccessFile`
  - `numPages: long`
  - `dbName: String`
  - `pageAccesses: int` - Statistics
- Methods:
  - `__init__(String)` / `__init__(String, int)` / `__init__(String, int, RandomAccessFile)`
  - `initialise(int)`: Create/open database
  - `reset()`: Wipe database
  - `getNumPages()`, `allocatePage()` → PageId: Page allocation
  - `deallocatePage(PageId)`: Mark page unused (no reuse)
  - `readPage(PageId, Page)`: Load from disk
  - `writePage(PageId, Page)`: Save to disk
  - `getPageAccesses()`: I/O count
- Exceptions: `IOException`, AssertionError

---

### 4. **buffer/** - Memory Management
Implements buffering and cache replacement strategy.

#### Classes:

**BufferFrame**
- Purpose: Wrapper for a single buffer cache slot
- Fields:
  - `content: Page`
  - `pageId: PageId` - Which page is in this frame
  - `isDirty: boolean` - Page modified
  - `pinned: int` - Pin count (prevent eviction)
- Methods:
  - `__init__()`
  - `isEmpty()`, `isDirty()`, `setDirty(boolean)`
  - `isPinned()`, `pin()`, `unpin()`: Pin management
  - `contains(PageId)`, `getPageId()`
  - `getPage()`: Returns page and increments pins
  - `setPage(PageId, Page)`: Load page into frame
- Exceptions: `BufferFrameDirtyException`, `AllBufferFramesPinnedException`

**Replacer (Interface)** - buffer/replacement/
- Purpose: Strategy for choosing eviction victim
- Methods:
  - `getName()` → String: Policy identifier
  - `choose(List<BufferFrame>)` → BufferFrame: Select victim
  - `notify(List<BufferFrame>, BufferFrame)`: Track access

**MruReplacer** implements Replacer
- Purpose: Most Recently Used eviction policy
- Methods:
  - `getName()` → "MRU"
  - `choose()`: Returns most recently used unpinned frame
  - `notify()`: Move accessed frame to end of list
  - `printStats()`

**BufferManager**
- Purpose: Main cache and buffer pool
- Fields:
  - `replacer: Replacer` - Eviction strategy
  - `frames: List<BufferFrame>` - Buffer pool
  - `maxFrames: int` - Pool size
  - `diskManager: DiskManager`
  - `cacheHits: int`, `pageAccesses: int` - Statistics
- Key Methods:
  - `__init__(int, Replacer, DiskManager)`
  - `getPage(PageId)` → Page: Get or load page
  - `getNewPage()` → PageId: Allocate new page
  - `pin(PageId)`, `unpin(PageId, boolean)`: Pin management
  - `markDirty(PageId)`: Mark for write-back
  - `flushDirty()` → int: Write dirty pages to disk
  - `flushPage(BufferFrame)`: Write single page
  - `getTotalDiskPages()` → long
  - **Private** `replaceFrameInBuffer()`: Handle miss
  - **Private** `findFrameByPageId()` → BufferFrame
  - **Private** `getBufferFrame()` → BufferFrame: Allocate new or evict
  - `getCacheHits()`, `getNumberOfPinnedPages()`, `getNumPageAccesses()`
  - `printStats()`
- Inner Exception: `BufferAccessException`

---

### 5. **access/read/** - Read Iterators
Sequential scan interfaces for table access.

#### Classes/Interfaces:

**AccessIterator (Abstract)** extends Iterator<Tuple>, Closeable
- Purpose: Generic iterator protocol for database scans
- Abstract Methods:
  - `hasNext()`, `next()`, `close()`
  - `getSchema()` → TupleDesc
  - `mark()`: Save current position
  - `reset()`: Restore to marked position
- Unsupported: `remove()`

**DataFileIterator (Abstract)** extends AccessIterator
- Purpose: Iterator over collection of pages in a file
- Fields:
  - `dataPageId: PageId` - Current page
  - `dataPage: DataPage`
  - `dataPageIterator: DataPageIterator` - Within-page iterator
  - `bufferManager: BufferManager`
  - `schema: TupleDesc`
  - `markedDataPageId: PageId`, `markedSlot: int` - Saved position
- Abstract Methods: `getDataPage(PageId)` → DataPage
- Concrete Methods:
  - `__init__(PageId, BufferManager, TupleDesc)`
  - `close()`: Unpin current page
  - `hasNext()`: Check current + next pages
  - `next()` → Tuple: Return next tuple
  - `mark()`: Save position
  - `reset()`: Restore position
- Exceptions: `BufferAccessException`, `NoSuchElementException`

**HeapFileIterator** extends DataFileIterator
- Purpose: Iterator for heap file pages
- Methods:
  - `__init__(PageId, BufferManager, TupleDesc)`
  - `getDataPage()`: Create HeapPage wrapper

**DataPageIterator** implements Iterator<Tuple>
- Purpose: Iterator within a single page
- Fields:
  - `dataPage: DataPage`
  - `slotNo: int` - Current slot
  - `schema: TupleDesc`
- Methods:
  - `__init__(DataPage, TupleDesc)`
  - `hasNext()`: Check if slot < recordCount
  - `next()` → Tuple: Read tuple from slot
  - `peekNext()` → Tuple: Look ahead without advancing
  - `getSlotNo()`, `setSlot(int)`: Position management
  - `remove()`: Unsupported

---

### 6. **access/write/** - Write Iterators
Sequential insertion interfaces for table writing.

#### Classes/Interfaces:

**AccessInserter (Abstract)** implements Closeable
- Purpose: Generic insertion protocol
- Abstract Methods:
  - `getSchema()` → TupleDesc
  - `insert(Tuple)`: Insert tuple
  - `canInsert()` → boolean: Check space available
  - `close()`
- Concrete Overload: `insert(Object...)`: Create tuple and insert
- Exceptions: `BufferAccessException`, `NoSuchElementException`

**DataFileInserter (Abstract)** extends AccessInserter
- Purpose: Inserter for file pages (with pagination)
- Fields:
  - `dataPageId: PageId`
  - `dataPage: DataPage`
  - `dataPageInserter: DataPageInserter`
  - `bufferManager: BufferManager`
  - `schema: TupleDesc`
- Abstract Methods: `getDataPage(PageId)` → DataPage
- Concrete Methods:
  - `__init__(PageId, BufferManager, TupleDesc)`
  - `insert(Tuple)`: Insert with automatic pagination
  - `canInsert()`: Check current or next page
  - `close()`: Unpin page

**HeapFileInserter** extends DataFileInserter
- Purpose: Inserter for heap files
- Methods:
  - `__init__(PageId, BufferManager, TupleDesc)`
  - `getDataPage()`: Create HeapPage wrapper

**DataPageInserter** extends AccessInserter
- Purpose: Inserter within a single page
- Fields:
  - `dataPage: DataPage`
  - `schema: TupleDesc`
- Methods:
  - `__init__(DataPage, TupleDesc)`
  - `insert(Tuple)`: Direct page insertion
  - `canInsert()` → boolean: Space check
  - `getSchema()`
  - `close()`: No-op

---

### 7. **join/** - Join Algorithm Implementations
Three join algorithms for combining relations.

#### Base Class:

**AbstractJoin (Abstract)** extends AccessIterator
- Purpose: Common join behavior
- Fields:
  - `schema: TupleDesc` - Output schema (union of inputs)
  - `left: AccessIterator`, `right: AccessIterator`
  - `leftColumn: int`, `rightColumn: int` - Join key indices
- Methods:
  - `__init__(AccessIterator, AccessIterator, JoinArgs)`
  - `joinTuple(Tuple, Tuple)` → Tuple: Combine rows
  - `close()`: Close both iterators
  - `getSchema()` → TupleDesc
- Abstract Methods: `hasNext()`, `next()`

#### Concrete Implementations:

**NestedLoopJoin** extends AbstractJoin
- Algorithm: Simple nested loops with O(M × N) page accesses
- Fields:
  - `next: Tuple` - Buffered result
  - `currentLeft: Tuple` - Current outer loop tuple
- Methods:
  - `__init__(AccessIterator, AccessIterator, JoinArgs)`
  - `hasNext()`: Find next matching pair
  - `next()` → Tuple: Return buffered result
  - **Private** `resetRight()`: Rewind inner loop
  - `mark()`, `reset()`, `close()`
- Exceptions: `BufferAccessException`, `NoSuchElementException`, `UnsupportedOperationException`

---

### 8. **parser/** - Query Parsing
Query language parsing and representation.

#### Classes:

**JoinArgs**
- Purpose: Encapsulate join parameters
- Fields:
  - `leftColumn: String`, `rightColumn: String`
  - `joinTable: String` - Table to join with
- Methods:
  - `__init__(String, String, String)`
  - `getLeftColumn()`, `getRightColumn()`, `getJoinTable()`
  - `toString()`

**Query**
- Purpose: Parsed query representation
- Syntax: `SELECT col1, col2 FROM table [JOIN table ON col = col];`
- Fields:
  - `tableName: String`
  - `projectedColumns: String[]`
  - `joinArgs: JoinArgs` (nullable)
- Methods:
  - `__init__(String, String[], JoinArgs)`
  - `getTableName()`, `getProjectedColumns()`, `getJoinArgs()`
  - `hasJoinArguments()` → boolean
  - `validate(DatabaseManager)` → String (error or null)
  - **Static** `generateQuery(String)` → Query (parses regex)
  - `toString()`

---

### 9. **execution/** - Query Processing
Main query execution engine.

#### Classes:

**QueryEngine**
- Purpose: REPL for SQL-like queries
- Fields: `dbms: DatabaseManager`
- Methods:
  - `__init__(DatabaseManager)`
  - `run()`: Interactive loop (accepts queries, supports TIME)
  - **Protected** `execute(Query)` → AccessIterator: Execute query
  - **Private** `checkQuery(Query)` → boolean: Validate
  - **Private** `printResult(AccessIterator)`: Display results
- Exceptions: `BufferAccessException`

---

### 10. **projection/** - Column Filtering
Apply column projection to result sets.

#### Classes:

**Projection** extends AccessIterator
- Purpose: Reduce tuple to projected columns only
- Fields:
  - `iterator: AccessIterator`
  - `schema: TupleDesc` - Projected schema
- Methods:
  - `__init__(AccessIterator, String[])`
  - `hasNext()`, `next()` → Tuple: Reduce row
  - `getSchema()`, `close()`
  - `mark()`, `reset()`: Unsupported
- Exceptions: `BufferAccessException`, `UnsupportedOperationException`

---

### 11. **demo/** - Example Usage
Demonstration of the system.

#### Classes:

**JoinDemo**
- Purpose: Example showing schema creation, data insertion, query execution
- Static Data: `STUDENT_ROWS_SMALL`, `TUTOR_ROWS_SMALL` (Object[][])
- Main Method:
  - Create schemas (students, tutors)
  - Insert test data
  - Launch QueryEngine REPL
- Static Method: `insertRows(HeapFile, Object[][])`

---

### 12. **test/** - Unit Tests

#### Test Classes:

**NestedLoopJoinTest**
- `testJoinSimple()`: Single page join
- `testJoinSimpleCommuted()`: Test with swapped relations
- `testJoinLarge()`: Multi-page join with cost verification
- Cost tolerance: 25%

#### Test Data Classes:

**TestData**
- `STUDENT_ROWS_SMALL` (4 rows): Michael, Jan, Roger, Rachael
- `STUDENT_ROWS` (146 rows): Large unsorted dataset

**SortedTestData**
- `STUDENT_ROWS_SMALL_SORTED`: 4 rows sorted by class
- `STUDENT_ROWS_SORTED_UNIQUE`: 55 unique entries sorted by class
- `STUDENT_ROWS_SORTED`: Full dataset sorted
- `TUTOR_ROWS_SORTED_UNIQUE`: 6 unique tutor entries
- `TUTOR_ROWS_SORTED`: Full sorted tutor dataset

#### Test Utilities:

**TestingUtils**
- `count(AccessIterator)` → int: Consume all tuples
- `count(HeapFile)` → int
- `countPages(HeapFile)` → int
- `insertRows(HeapFile, Object[][])`: Batch insert
- `checkGetRecord(AccessIterator, Object...)`: Assert next row matches

---

## Data Type System

### Type Hierarchy:
```
Type (enum)
├── STRING    → 2 + MAX_STRING_LENGTH (17 bytes)
├── DOUBLE    → 8 bytes
├── INTEGER   → 4 bytes
└── BOOLEAN   → 1 byte
```

### Serialization Format (Page Level):
- **STRING**: 2-byte length prefix + ASCII bytes
- **DOUBLE**: 8-byte IEEE 754
- **INTEGER**: 4-byte big-endian
- **BOOLEAN**: 1 byte (0 = false, 1 = true)

---

## Storage Layout

### Page Structure (1024 bytes):
```
Header (Variable Length):
├── [Special fields for page type - see DataPage/HeaderPage]
├── [Type-specific metadata]

Data Section:
└── [Records/pointers starting at RECORD_START_POS]
```

### DataPage Header:
```
Offset 0:  Previous Page ID (4 bytes)
Offset 4:  Next Page ID (4 bytes)
Offset 8:  Record Count (4 bytes)
Offset 12: Relation Name (2 + 30 bytes)
Offset 44: Record Data Starts
```

### HeaderPage Header:
```
Offset 0:  Next Header Page ID (4 bytes)
Offset 4:  Max Pointers (4 bytes)
Offset 8:  Pointer Entries (repeating, 36 bytes each)
           ├── Page ID (4 bytes)
           ├── Name Length (2 bytes)
           └── Name (30 bytes)
```

### Linked List Structure:
- Pages form linked lists (doubly-linked for data pages)
- Header pages chain together for large catalogs
- Enables sequential scans without full catalog

---

## Key Algorithms

### Nested Loop Join
```
for each outer tuple r:
  for each inner tuple s:
    if r.joinKey == s.joinKey:
      emit join(r, s)
```
- Cost: M + (M × N × (1/B)) where M=outer pages, N=inner pages, B=inner block size
- Constructor: O(1)
- First Result: O(M pages + scan to first match)

### Block Nested Loop Join
```
for each block Br of outer (size B pages):
  for each block Bs of inner (size B pages):
    for each tuple r in Br:
      for each tuple s in Bs:
        if r.joinKey == s.joinKey:
          emit join(r, s)
```
- Cost: M + (⌈M/B⌉ × N) where B = block size in pages
- Benefit: Reduces inner scan repetitions
- Reset: Required between outer blocks

### Sort-Merge Join
```
Assumes both inputs pre-sorted on join keys
Pointers advance: left matches right when keys equal
Handles duplicates by:
  - Saving right pointer at first match
  - Continuing right until key changes
  - Restoring pointer for next left tuple
```
- Cost: M + N (linear in sorted input size)
- Requires: Pre-sorted inputs (NOT implemented - assumed)
- Memory: O(duplicate count) for nested matches

---

## Dependencies

### External Java Libraries:
- **junit** (4.x+): Unit testing framework
- **java.io**: File I/O (RandomAccessFile)
- **java.nio**: ByteBuffer for binary serialization
- **java.util**: Collections, generics, Iterator protocol
  - HashMap, ArrayList, LinkedList, Scanner
  - Comparator, Iterator, Closeable
- **java.util.regex**: Pattern matching for query parsing
- **java.nio.charset**: StandardCharsets.US_ASCII for strings

### Internal Dependencies (Compile Graph):

```
demo/JoinDemo
  ├── execution/QueryEngine
  ├── global/DatabaseManager
  ├── heap/{HeapFile, TupleDesc}
  └── access/write/AccessInserter

execution/QueryEngine
  ├── heap/HeapFile
  ├── parser/Query
  ├── projection/Projection
  └── join/NestedLoopJoin

join/{NestedLoopJoin, BlockNestedLoopJoin, SortMergeJoin}
  ├── join/AbstractJoin
  └── access/read/AccessIterator

heap/{HeapFile, HeapPage}
  ├── heap/TupleDesc
  ├── access/read/{AccessIterator, HeapFileIterator}
  ├── access/write/{AccessInserter, HeapFileInserter}
  ├── buffer/BufferManager
  └── disk/{PageId, HeaderPage}

buffer/BufferManager
  ├── buffer/BufferFrame
  ├── buffer/replacement/Replacer
  ├── disk/{DiskManager, Page, PageId}
  └── global/DatabaseConstants

disk/DiskManager
  └── disk/{HeaderPage, PageId, Page, DataPage}
```

---

## Key Data Flow Patterns

### Reading (Scan):
```
HeapFile.iterator()
  ↓
HeapFileIterator(firstPageId)
  ↓
DataFileIterator: manages page transitions
  ↓
DataPageIterator: manages within-page iteration
  ↓
Tuple (with pageId, slotId set)
```

### Writing (Insert):
```
HeapFile.inserter()
  ↓
HeapFileInserter(firstPageId)
  ↓
DataFileInserter: auto-pagination when page full
  ↓
DataPageInserter: direct page insertion
  ↓
Page marked dirty and pinned
```

### Query Execution:
```
Query.generateQuery(string)
  ↓
Query.validate(dbms)
  ↓
QueryEngine.execute(query)
  ├─ Load left table: dbms.getHeapFile(name).iterator()
  ├─ [If join] Load right table & create join operator
  ├─ [Apply projection] Wrap with Projection operator
  └─ Return result AccessIterator
  ↓
Consumer iterates: while(hasNext()) { next() }
```

---

## Important Design Patterns

1. **Iterator Pattern**: AccessIterator throughout for lazy evaluation
2. **Abstract Base Classes**: AbstractJoin, AccessIterator, AccessInserter for extending
3. **Strategy Pattern**: Replacer interface for buffer replacement policies
4. **Factory Pattern**: Implicit in DatabaseManager component creation
5. **Linked List**: Pages chain together for sequential access
6. **Enum with Abstract Methods**: Type enum uses abstract `getLen()`
7. **Pin/Unpin Reference Count**: Prevent eviction of in-use pages

---

## Error Handling

- **Checked Exceptions:**
  - `IOException`: Disk I/O failures
  - `BufferAccessException`: Buffer operation failures
  
- **Unchecked:**
  - `NoSuchElementException`: Iterator exhausted
  - `UnsupportedOperationException`: Unimplemented methods
  - `AssertionError`: Precondition violations
  - `RuntimeException`: Wrapping checked exceptions in unchecked contexts
  
- **Custom Exceptions:**
  - `BufferFrame.BufferFrameDirtyException`: Attempt to evict dirty frame by reference
  - `BufferFrame.AllBufferFramesPinnedException`: No unpinned frames for eviction
  - `DatabaseManager.ComponentsNotInitialisedError`: Before initialization

---

## Configuration & Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| PAGE_SIZE | 1024 | Fixed page size |
| MAX_BUFFER_FRAMES | 32 | Buffer pool size |
| MAX_TABLE_NAME_LENGTH | 30 | Schema name limit |
| MAX_STRING_LENGTH | 15 | String column limit |
| FIRST_PAGE_ID | 0 | Header page location |
| INVALID_PAGE_ID | -1 | Null page marker |

---

## Testing Strategy

- **JUnit 4** framework
- **Timeout Rule**: 1000ms per test method (except sort-merge: 1000ms)
- **Cost Tolerance**: 10-50% above theoretical minimum
- **Test Dataset Sizes:**
  - Small: 4-6 rows per relation
  - Large: 146 students × 6 tutors (8089 results)
  - Unique: 55 students × 6 tutors (55 results)

---

## Files Summary by Directory

| Directory | Files | Purpose |
|-----------|-------|---------|
| src/global/ | Type.java, DatabaseConstants.java, Catalog.java, DatabaseManager.java | DB config & initialization |
| src/heap/ | Tuple.java, TupleDesc.java, HeapFile.java, HeapPage.java | Row & schema structures |
| src/disk/ | Page.java, DataPage.java, HeaderPage.java, PageId.java, DiskManager.java | Persistence layer |
| src/buffer/ | BufferFrame.java, BufferManager.java, Replacer.java, MruReplacer.java | Memory cache |
| src/access/read/ | AccessIterator.java, DataFileIterator.java, HeapFileIterator.java, DataPageIterator.java | Read access |
| src/access/write/ | AccessInserter.java, DataFileInserter.java, HeapFileInserter.java, DataPageInserter.java | Write access |
| src/join/ | AbstractJoin.java, NestedLoopJoin.java, BlockNestedLoopJoin.java, SortMergeJoin.java, JoinComparator.java | Join algorithms |
| src/parser/ | Query.java, JoinArgs.java | Query parsing |
| src/execution/ | QueryEngine.java | Query REPL |
| src/projection/ | Projection.java | Column filtering |
| src/demo/ | JoinDemo.java | Example usage |
| test/join/ | NestedLoopJoinTest.java, BlockNestedLoopJoinTest.java, SortMergeJoinTest.java | Join tests |
| test/join/data/ | TestData.java, SortedTestData.java | Test fixtures |
| test/utilities/ | TestingUtils.java | Test helpers |

---

## Notes for Python Conversion

1. **ByteBuffer replacement**: Python `struct` module or `bytes.io`
2. **RandomAccessFile**: Python's built-in `open()` with 'r+b' mode
3. **Interface/Abstract Class**: Python ABC (abc.ABC, @abstractmethod)
4. **Generics**: Use type hints (PEP 585+ or typing module)
5. **Resource management**: with statements instead of try-with-resources
6. **Enums**: Use Python's `Enum` class
7. **LinkedList**: Use collections.deque or list for simple cases
8. **HashMap**: Use dict with object keys (ensure __hash__, __eq__)
9. **Inheritance**: Direct Python class inheritance works identically

