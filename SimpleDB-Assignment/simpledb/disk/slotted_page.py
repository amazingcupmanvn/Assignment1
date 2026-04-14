"""
Slotted Page base class for slotted page architecture.
"""

from simpledb.main.database_constants import DatabaseConstants
from simpledb.disk.page import Page
from simpledb.heap.page_id import PageId


class SlottedPage(Page):
    """Base class for pages using slotted page architecture."""

    # Header field positions
    MAGIC_POS = 0
    VERSION_TYPE_POS = 2
    NEXT_PAGE_POS = 4
    NUM_SLOTS_POS = 8
    FREE_START_POS = 10
    FREE_END_POS = 12
    NAME_LENGTH_POS = 14
    NAME_STRING_POS = 16
    SLOT_DIR_START = NAME_STRING_POS + DatabaseConstants.MAX_TABLE_NAME_LENGTH  # 16 + 32 = 48

    def __init__(self, data: bytes = None):
        super().__init__(data)

    def initialise(self, page_type) -> None:
        """Initialize an empty SlottedPage to default values."""
        self.set_magic(DatabaseConstants.PAGE_MAGIC)
        self.set_version_type(page_type)
        self.set_next_page(PageId(DatabaseConstants.INVALID_PAGE_ID))
        self.set_num_slots(0)
        self.set_free_start(self.SLOT_DIR_START)
        self.set_free_end(DatabaseConstants.PAGE_SIZE)
        self.set_schema_name("")

    def get_magic(self) -> int:
        """Get the magic identifier."""
        return self.get_short_value(self.MAGIC_POS)

    def set_magic(self, magic: int) -> None:
        """Set the magic identifier."""
        self.set_short_value(magic, self.MAGIC_POS)

    def get_version_type(self) -> int:
        """Get the version/type."""
        return self.get_short_value(self.VERSION_TYPE_POS)

    def set_version_type(self, version_type: int) -> None:
        """Set the version/type."""
        self.set_short_value(version_type, self.VERSION_TYPE_POS)

    def get_next_page(self) -> PageId:
        """Get the next page ID."""
        page_id = self.get_integer_value(self.NEXT_PAGE_POS)
        return PageId(page_id)

    def set_next_page(self, page_id: PageId) -> None:
        """Set the next page ID."""
        if page_id is None:
            raise RuntimeError("Page ID is NULL")
        self.set_integer_value(page_id.get(), self.NEXT_PAGE_POS)
        
    def get_num_slots(self) -> int:
        """Get the number of slots."""
        return self.get_short_value(self.NUM_SLOTS_POS)

    def set_num_slots(self, num_slots: int) -> None:
        """Set the number of slots."""
        self.set_short_value(num_slots, self.NUM_SLOTS_POS)

    def get_free_start(self) -> int:
        """Get the start of free space."""
        return self.get_short_value(self.FREE_START_POS)

    def set_free_start(self, free_start: int) -> None:
        """Set the start of free space."""
        self.set_short_value(free_start, self.FREE_START_POS)

    def get_free_end(self) -> int:
        """Get the end of free space."""
        return self.get_short_value(self.FREE_END_POS)

    def set_free_end(self, free_end: int) -> None:
        """Set the end of free space."""
        self.set_short_value(free_end, self.FREE_END_POS)

    def get_schema_name(self) -> str:
        """Get the page content's schema name."""
        length = self.get_short_value(self.NAME_LENGTH_POS)
        return self.data[self.NAME_STRING_POS:self.NAME_STRING_POS + length].decode('ascii')

    def set_schema_name(self, name: str) -> None:
        """Set the schema name of entries on this page."""
        name_bytes = name.encode('ascii')
        length = len(name_bytes)
        if length > DatabaseConstants.MAX_TABLE_NAME_LENGTH:
            raise ValueError(f"Name too long: {length} > {DatabaseConstants.MAX_TABLE_NAME_LENGTH}")
        self.set_short_value(length, self.NAME_LENGTH_POS)
        self.data[self.NAME_STRING_POS:self.NAME_STRING_POS + DatabaseConstants.MAX_TABLE_NAME_LENGTH] = b'\x00' * DatabaseConstants.MAX_TABLE_NAME_LENGTH
        self.data[self.NAME_STRING_POS:self.NAME_STRING_POS + length] = name_bytes

    def get_slot_offset(self, slot_id: int) -> int:
        """Get the offset for the given slot ID."""
        if slot_id >= self.get_num_slots():
            raise IndexError(f"Slot ID {slot_id} out of range")
        slot_pos = self.SLOT_DIR_START + slot_id * DatabaseConstants.SLOT_ENTRY_SIZE
        return self.get_short_value(slot_pos)

    def set_slot_offset(self, slot_id: int, offset: int) -> None:
        """Set the offset for the given slot ID."""
        if slot_id >= self.get_num_slots():
            raise IndexError(f"Slot ID {slot_id} out of range")
        slot_pos = self.SLOT_DIR_START + slot_id * DatabaseConstants.SLOT_ENTRY_SIZE
        self.set_short_value(offset, slot_pos)

    def find_free_slot(self) -> int:
        """Find the first free slot (offset == 0)."""
        num_slots = self.get_num_slots()
        for slot_id in range(num_slots):
            if self.get_slot_offset(slot_id) == 0:
                return slot_id
        return -1  # No free slot

    def allocate_slot(self, record_size: int) -> int:
        """Allocate a slot for a record of the given size."""
        free_end = self.get_free_end()
        if free_end - record_size - DatabaseConstants.SLOT_ENTRY_SIZE < self.get_free_start():
            raise OverflowError("Not enough free space")

        free_slot = self.find_free_slot()
        if free_slot == -1:
            # Need to add a new slot
            free_slot = self.get_num_slots()
            self.set_num_slots(free_slot + 1)

        offset = free_end - record_size
        self.set_slot_offset(free_slot, offset)
        self.set_free_end(offset)
        return free_slot


# specified in plan, but unused and not sure the current code actually works; may need to be rethought if we want to implement it
#    def write_record_at_end(self, record_bytes: bytes) -> int:
#        """Write record bytes at the end of free space and return the slot ID."""
#        # Record format: length (2 bytes) + data
#        record_size = 2 + len(record_bytes)
#        slot_id = self.allocate_slot(record_size)
#        offset = self.get_slot_offset(slot_id)
#        self.set_short_value(len(record_bytes), offset)
#        self.data[offset + 2:offset + 2 + len(record_bytes)] = record_bytes  # not sure this is right...
#        return slot_id
#
#    def get_record_bytes(self, slot_id: int) -> bytes:
#        """Get the record bytes for the given slot ID."""
#        offset = self.get_slot_offset(slot_id)
#        if offset == 0:
#            raise ValueError(f"Slot {slot_id} is empty")
#        length = self.get_short_value(offset)
#        return bytes(self.data[offset + 2:offset + 2 + length])   # not sure this is right...