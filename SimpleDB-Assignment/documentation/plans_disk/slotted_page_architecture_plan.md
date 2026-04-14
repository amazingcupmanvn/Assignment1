## Plan: Generalize Page Layout to Slotted Page Architecture

Refactor the existing page classes (Page, HeaderPage, DataPage) into a slotted page architecture where HeaderPage, DataPage, and IndexPage are specializations of a base SlottedPage class. The general page includes a standardized header with magic identifier, version/type, page links, free space pointers, name record, and slot directory for managing variable-length records stored from the page end.

**Steps**

1. Define slotted page constants in `simpledb/main/database_constants.py`
   - Change MAX_TABLE_NAME_LENGTH = 32 (from 30)
   - Add SLOT_PAGE_MAGIC = 0x3404
   - Add SLOT_PAGE_HEADER_SIZE (calculate based on fixed header fields)
   - Add SLOT_ENTRY_SIZE = 2 (bytes per slot entry)
   - Add MAX_SLOT_ENTRIES (based on page size)
   - Add page type constants: HEADER_PAGE_TYPE = 0, DATA_PAGE_TYPE = 1, INDEX_NODE_TYPE = 2, INDEX_LEAF_TYPE = 3

1.5 Update `simpledb/disk/page.py` to add `get_short_value(offset: int) -> int` and `set_short_value(value: int, offset: int) -> None` methods for reading/writing 2-byte short values using struct '>H' (big-endian unsigned short). These methods will be used for accessing header fields like offsets, num_slots, etc., alongside existing primitive access methods.

2. Create `simpledb/disk/slotted_page.py` base class
   - Inherit from Page
   - Implement header layout: magic (0-1), version/type (2-3), next_page (4-7), num_slots (8-9), free_start (10-11), free_end (12-13), schema_name_length (14-15), schema_name_string (16 to 15+MAX_TABLE_NAME_LENGTH)
   - Slot directory starts at SLOT_DIR_START = 16 + MAX_TABLE_NAME_LENGTH
   - Implement methods: get/set_magic, get/set_version_type, get/set_next_page, get/set_free_start/end, get/set_schema_name, get/set_num_slots
   - Implement slot management: get_slot_offset(slot_id), set_slot_offset(slot_id, offset), find_free_slot(), allocate_slot(record_size)
   - Implement record storage: write_record_at_end(record_bytes), get_record_bytes(slot_id)

3. Refactor `simpledb/disk/header_page.py` to inherit from SlottedPage
   - Set version_type to HEADER_PAGE_TYPE
   - Rename all file_entry methods to catalog_entry:
     - `set_file_entry` → `set_catalog_entry`
     - `get_file_entry` → `get_catalog_entry`
     - `get_file_entry_static` → `get_catalog_entry_static`
     - `set_file_entry_static` → `set_catalog_entry_static`
   - Adapt existing methods to use slotted storage for catalog entries
   - Each catalog entry (page_id + name) stored as record, slot points to it
   - Update initialise, set/get_catalog_entry, etc. to use slots

4. Refactor `simpledb/disk/data_page.py` to inherit from SlottedPage
   - Set version_type to DATA_PAGE_TYPE
   - Adapt tuple storage to use slots instead of fixed positions
   - Update insert_record, get_record to use slot allocation and retrieval
   - Maintain relation_name in header
   - Remove prev_page methods (get_previous_page_id, set_previous_page_id)

5. Create `simpledb/disk/index_page.py` as new specialization (base class for index pages)
   - Inherit from SlottedPage
   - Will be further specialized into INDEX_NODE_TYPE and INDEX_LEAF_TYPE subclasses
   - Implement index entry storage (key + value pointers)
   - Add methods for index operations (insert, search, etc.)

6. Update any dependent classes
   - Refactor `simpledb/heap/heap_file.py` to use new static method names: `get_catalog_entry_static` and `set_catalog_entry_static`
   - Update `simpledb/access/write/heap_file_inserter.py` to not set prev_page on new pages
   - Check buffer_manager, heap_file, etc. for page type assumptions
   - Ensure page creation uses appropriate subclasses

**Relevant files**
- `simpledb/main/database_constants.py` — Add slotted page constants
- `simpledb/disk/page.py` — Add short value methods
- `simpledb/disk/slotted_page.py` — New base class for slotted pages
- `simpledb/disk/header_page.py` — Rename methods and refactor to SlottedPage
- `simpledb/disk/data_page.py` — Refactor to SlottedPage
- `simpledb/disk/index_page.py` — New index page specialization
- `simpledb/heap/heap_file.py` — Update calls to renamed static methods
- `simpledb/access/write/heap_file_inserter.py` — Remove prev_page setting

**Verification**
1. Run existing tests to ensure no regressions in header and data page functionality
2. Verify catalog entry methods work with new names and slotted storage
3. Create unit tests for SlottedPage base methods (slot allocation, record storage)
4. Test page type identification and header parsing
5. Verify free space management prevents overflows
6. Test tuple insertion/deletion on data pages with slotted layout

**Decisions**
- Catalog entries renamed from file_entry to catalog_entry for clarity
- Page types: 0=HEADER_PAGE_TYPE, 1=DATA_PAGE_TYPE, 2=INDEX_NODE_TYPE, 3=INDEX_LEAF_TYPE
- MAX_TABLE_NAME_LENGTH = 32 bytes (gives a 48 byte long page header which is easy to compute with and easy to read in hexdump)
- Name record: stored as (2-byte length, fixed-length 32-byte string) tuple
- Slot directory: array of 2-byte offsets, grows from header towards free space
- Deleted slots marked with offset 0; slot directory entries do not change after creation
- Records: stored from page end towards free space, variable length
- Free space: managed between slot dir end and records start
- No prev_page pointer; backward traversal handled by restarting from first page if needed
- Short values (2 bytes) used for all offset and count fields in slotted page header and slot directory to optimize space usage