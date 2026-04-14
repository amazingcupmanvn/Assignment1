"""
Header Page class for the database catalog.
"""

from simpledb.disk.slotted_page import SlottedPage
from simpledb.heap.page_id import PageId
from simpledb.main.database_constants import DatabaseConstants


class HeaderPage(SlottedPage):
    """Represents the header page containing the catalog."""

    def __init__(self, page: SlottedPage = None):
        """Initialize a HeaderPage."""
        if page is None:
            super().__init__()
        else:
            super().__init__(page.get_data())

    def initialise(self) -> None:
        """Initialize the HeaderPage to default values."""
        super().initialise(DatabaseConstants.HEADER_PAGE_TYPE)
        self.set_schema_name("SimpleDB Catalog")

    def get_entry(self, slot_number: int, first_page: PageId) -> str:
        """Get the (pageId, entryText) entry associated with the catalog slot number."""
        assert 0 <= slot_number < self.get_num_slots()
        offset = self.get_slot_offset(slot_number)
        if offset == 0:
            first_page.set(DatabaseConstants.INVALID_PAGE_ID)
            return ""
        first_page.set(self.get_integer_value(offset))
        return self.get_string_value(offset + PageId.SIZE)

    def add_entry(self, entry_name: str, entry_page_id: PageId) -> int:
        """Create a catalog entry in this header page of (pageId, entryText)."""
        assert len(entry_name) <= DatabaseConstants.MAX_TABLE_NAME_LENGTH

        try:
            record_size = PageId.SIZE + 2 + DatabaseConstants.MAX_TABLE_NAME_LENGTH
            slot_number = self.allocate_slot(record_size)
            offset = self.get_slot_offset(slot_number)
            self.set_integer_value(entry_page_id.get(), offset)
            self.set_string_value(entry_name, offset + PageId.SIZE)
            return slot_number
        except OverflowError:
            return -1  # indicates failure to insert

    @staticmethod
    def find_catalog_entry(buffer_manager, search_name: str) -> PageId:
        """Get the file entry from the header page."""
        if len(search_name) > DatabaseConstants.MAX_TABLE_NAME_LENGTH:
            raise AssertionError("Catalog entry name cannot be longer than %d characters" % DatabaseConstants.MAX_TABLE_NAME_LENGTH)

        current_page_id = PageId(DatabaseConstants.FIRST_PAGE_ID)
        entry_page_id = PageId()
        
        while current_page_id.is_valid():
            hpage = HeaderPage(buffer_manager.get_page(current_page_id))
            num_records = hpage.get_num_slots()
            for i in range(num_records):
                entry_name = hpage.get_entry(i, entry_page_id)
                if entry_name.capitalize() == search_name.capitalize() and entry_page_id.is_valid():
                    buffer_manager.unpin(current_page_id, False)
                    return entry_page_id
            
            next_page_id = hpage.get_next_page()
            buffer_manager.unpin(current_page_id, False)
            current_page_id = next_page_id
        
        return PageId(DatabaseConstants.INVALID_PAGE_ID)

    @staticmethod
    def insert_catalog_entry(buffer_manager, entry_name: str, entry_page_id: PageId) -> None:
        """Put a file entry in the header page."""
        if len(entry_name) > DatabaseConstants.MAX_TABLE_NAME_LENGTH:
            raise AssertionError("Catalog entry name cannot be longer than %d characters" % DatabaseConstants.MAX_TABLE_NAME_LENGTH)

        if not (entry_page_id.is_valid() and entry_page_id.get() < buffer_manager.get_total_disk_pages()):
            raise AssertionError("Expects catalog entry's PageId to exist")

        if HeaderPage.find_catalog_entry(buffer_manager, entry_name).is_valid():
            raise AssertionError("Catalog entry %s already exists" % entry_name)

        current_page_id = PageId(DatabaseConstants.FIRST_PAGE_ID)        
        while current_page_id.is_valid():
            hpage = HeaderPage(buffer_manager.get_page(current_page_id))
            if hpage.add_entry(entry_name, entry_page_id) != -1:
                buffer_manager.unpin(current_page_id, True)
                buffer_manager.flush_dirty()
                return            
            # no space on the current header page, move to the next one
            next_page_id = hpage.get_next_page()
            buffer_manager.unpin(current_page_id, False)
            current_page_id = next_page_id
        
        # If we reach here, no space on any header page, so need to create new page
        new_page_id = buffer_manager.get_new_page()
        new_page = HeaderPage(buffer_manager.get_page(new_page_id))
        new_page.initialise()
        new_page.add_entry(entry_name, entry_page_id)
        buffer_manager.unpin(new_page_id, True)
        buffer_manager.flush_dirty()
