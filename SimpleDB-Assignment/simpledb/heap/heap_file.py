"""
HeapFile - a collection of unordered pages containing tuples.
"""

from simpledb.main.catalog.tuple_desc import TupleDesc
from simpledb.heap.page_id import PageId
from simpledb.heap.heap_page import HeapPage
from simpledb.buffer.buffer_manager import BufferManager, BufferAccessException
from simpledb.disk.header_page import HeaderPage
from simpledb.main.database_constants import DatabaseConstants
import time
import random


class HeapFile:
    """Represents a collection of unordered pages containing tuples."""

    def __init__(self, schema: TupleDesc, relation_name: str = None, buffer_manager: BufferManager = None):
        """Initialize a HeapFile."""
        self.schema = schema
        self.buffer = buffer_manager
        self.relation_name = relation_name
        self.first_page_id = None
        
        if relation_name is not None and buffer_manager is not None:
            # Persistent HeapFile
            self.first_page_id = HeaderPage.find_catalog_entry(buffer_manager, relation_name)
            if not self.first_page_id.is_valid():
                self.first_page_id = buffer_manager.get_new_page()
                HeaderPage.insert_catalog_entry(buffer_manager, relation_name, self.first_page_id)
                first_page = HeapPage(buffer_manager.get_page(self.first_page_id), schema)
                first_page.initialise(relation_name)
                buffer_manager.unpin(self.first_page_id, True)
        else:
            # Temporary HeapFile
            ts = str(int(time.time() * 1000))
            rnd = str(100 + (int(random.random() * 100) % 100))
            self.relation_name = "tmp" + ts[-min(DatabaseConstants.MAX_TABLE_NAME_LENGTH + 6, len(ts)):]
            self.first_page_id = buffer_manager.get_new_page()
            first_page = HeapPage(buffer_manager.get_page(self.first_page_id), schema)
            first_page.initialise(self.relation_name)
            buffer_manager.unpin(self.first_page_id, True)

    def iterator(self):
        """Get an iterator over tuples in this file."""
        from simpledb.access.read.heap_file_iterator import HeapFileIterator
        return HeapFileIterator(self.buffer, self.first_page_id, self.schema)

    def inserter(self):
        """Get an inserter for this file."""
        from simpledb.access.write.heap_file_inserter import HeapFileInserter
        return HeapFileInserter(self.buffer, self.first_page_id, self.schema)

    def get_schema(self) -> TupleDesc:
        """Get the schema of this heap file."""
        return self.schema

    def is_empty(self) -> bool:
        """Check if this heap file is empty."""
        if self.first_page_id is None or self.first_page_id.get() == DatabaseConstants.INVALID_PAGE_ID:
            return True
        else:
            first_page = HeapPage(self.buffer.get_page(self.first_page_id),self.schema)
            records_on_1st_pg = first_page.get_record_count()
            self.buffer.unpin(self.first_page_id, False)
            return records_on_1st_pg == 0

    def print_stats(self) -> str:
        """Print statistics about this heap file."""
        return f"Relation {self.relation_name}, firstPageId {self.first_page_id.get()}, schema {hash(self.schema)}"
