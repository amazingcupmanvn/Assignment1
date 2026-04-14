"""
Database constants definition.
"""

class DatabaseConstants:
    """Static constants for the database."""
    DEFAULT_DB_NAME = "data3404_database.db"
    PAGE_SIZE = 1024
    INVALID_PAGE_ID = -1
    FIRST_PAGE_ID = 0
    MAX_TABLE_NAME_LENGTH =32
    MAX_STRING_LENGTH = 14
    MAX_BUFFER_FRAMES = 32

    # Slotted page constants
    PAGE_MAGIC = 0x3404
    PAGE_HEADER_SIZE = 48  # magic(2) + version/type(2) + next_page(4) + num_slots(2) + free_start(2) + free_end(2) + name_length(2) + name_string(32) = 48
    SLOT_ENTRY_SIZE = 2
    MAX_SLOT_ENTRIES = (PAGE_SIZE - PAGE_HEADER_SIZE) // SLOT_ENTRY_SIZE  # 488

    # Page type constants
    HEADER_PAGE_TYPE = 0
    DATA_PAGE_TYPE = 1
    INDEX_NODE_TYPE = 2
    INDEX_LEAF_TYPE = 3
