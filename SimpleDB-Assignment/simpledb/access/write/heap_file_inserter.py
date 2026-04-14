"""
Heap File Inserter for writing tuples to a HeapFile.
"""

from simpledb.access.write.access_inserter import AccessInserter
from simpledb.buffer.buffer_manager import BufferManager
from simpledb.main.catalog.tuple_desc import TupleDesc
from simpledb.heap.page_id import PageId
from simpledb.heap.tuple import Tuple
from simpledb.heap.heap_page import HeapPage


class HeapFileInserter(AccessInserter):
    """Inserter for writing tuples to a HeapFile."""

    def __init__(self, buffer_manager: BufferManager, first_page_id: PageId, schema: TupleDesc):
        """Initialize the HeapFileInserter."""
        self.buffer_manager = buffer_manager
        self.schema = schema
        self.current_page_id = PageId(first_page_id.get())

    def get_schema(self) -> TupleDesc:
        """Get the schema of tuples."""
        return self.schema

    def insert(self, row: list) -> None:
        """Insert a row into the heap file."""
        tuple_obj = Tuple(self.schema, row)
        
        while self.current_page_id.is_valid():
            page = self.buffer_manager.get_page(self.current_page_id)
            heap_page = HeapPage(page, self.schema)
            
            if heap_page.insert_record(tuple_obj):
                self.buffer_manager.unpin(self.current_page_id, True)
                return
            
            next_page_id = heap_page.get_next_page()
            self.buffer_manager.unpin(self.current_page_id, False)
            
            if not next_page_id.is_valid():
                # Create new page
                new_page_id = self.buffer_manager.get_new_page()
                new_page = HeapPage(self.buffer_manager.get_page(new_page_id), self.schema)
                new_page.initialise(heap_page.get_schema_name())
                
                # Link pages
                self.buffer_manager.unpin(new_page_id, True)
                old_page = HeapPage(self.buffer_manager.get_page(self.current_page_id), self.schema)
                old_page.set_next_page(new_page_id)
                self.buffer_manager.unpin(self.current_page_id, True)
                
                self.current_page_id = new_page_id
            else:
                self.current_page_id = next_page_id

    def close(self) -> None:
        """Close the inserter."""
        pass
