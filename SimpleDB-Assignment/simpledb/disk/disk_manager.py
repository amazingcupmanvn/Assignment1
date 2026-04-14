"""
Disk Manager for the database.
"""

import os
from simpledb.main.database_constants import DatabaseConstants
from simpledb.heap.page_id import PageId
from simpledb.disk.page import Page
from simpledb.disk.header_page import HeaderPage


class DiskManager:
    """Manages access to the disk."""

    def __init__(self, db_name: str, num_pages: int = 1, tmp_file = None):
        """Open/create a database with the given name."""
        self.db_name = db_name
        self.num_pages = 0
        self.page_accesses = 0

        # Check if file exists or is already opened as temp file
        if tmp_file is not None:
            self.db_file = tmp_file
        else:
            file_exists = os.path.exists(db_name)
            self.db_file = open(db_name, 'r+b' if file_exists else 'w+b')
        
        self._initialise(num_pages)

    def _initialise(self, num_pages: int) -> None:
        """Initialize the database."""
        self.db_file.seek(0, 2)  # Seek to end to check file size; offset 0 from end (from_what=2 means end of file)
        file_size = self.db_file.tell()
        
        if file_size == 0:
            # Initialize first header page
            self.num_pages = num_pages
            hpage = HeaderPage()
            hpage.initialise()
            self._write_page(PageId(DatabaseConstants.FIRST_PAGE_ID), hpage)
            # Extend file to num_pages length
            self.db_file.seek(num_pages * DatabaseConstants.PAGE_SIZE - 1)
            self.db_file.write(b'\0')
        
        self.db_file.seek(0, 2)
        self.num_pages = self.db_file.tell() // DatabaseConstants.PAGE_SIZE
        self.page_accesses = 0

    def reset(self) -> None:
        """Reset the file representing the database."""
        try:
            self.db_file.close()
        except:
            pass
        
        if os.path.exists(self.db_name):
            os.remove(self.db_name)
        
        self.db_file = open(self.db_name, 'w+b')
        self._initialise(1)

    def get_num_pages(self) -> int:
        """Get the number of pages in the database."""
        return self.num_pages

    def allocate_page(self) -> PageId:
        """Allocate a page on disk."""
        page_id = PageId(self.num_pages)
        self.num_pages += 1
        return page_id

    def deallocate_page(self, page_id: PageId) -> None:
        """Deallocate the page associated with the given page ID."""
        hpage = HeaderPage()
        try:
            self._read_page(page_id, hpage)
            hpage.set_num_pointers(DatabaseConstants.INVALID_PAGE_ID)
            self._write_page(page_id, hpage)
            page_id.set(DatabaseConstants.INVALID_PAGE_ID)
        except:
            pass

    def read_page(self, page_id: PageId, page: Page) -> None:
        """Read the page from disk."""
        assert page_id.is_valid() and page_id.get() < self.num_pages
        self._read_page(page_id, page)

    def _read_page(self, page_id: PageId, page: Page) -> None:
        """Internal method to read page from disk."""
        self.db_file.seek(page_id.get() * DatabaseConstants.PAGE_SIZE)
        data = self.db_file.read(DatabaseConstants.PAGE_SIZE)
        if len(data) == DatabaseConstants.PAGE_SIZE:
            page.set_data(data)
        self.page_accesses += 1

    def write_page(self, page_id: PageId, page: Page) -> None:
        """Write the page to disk."""
        assert page_id.is_valid() and page_id.get() < self.num_pages
        self._write_page(page_id, page)

    def _write_page(self, page_id: PageId, page: Page) -> None:
        """Internal method to write page to disk."""
        self.db_file.seek(page_id.get() * DatabaseConstants.PAGE_SIZE)
        self.db_file.write(page.get_data())
        self.db_file.flush()
        self.page_accesses += 1

    def get_page_accesses(self) -> int:
        """Get the number of disk reads/writes that have taken place."""
        return self.page_accesses

    def __del__(self):
        """Close the database file."""
        if hasattr(self, 'db_file'):
            try:
                self.db_file.close()
            except:
                pass
