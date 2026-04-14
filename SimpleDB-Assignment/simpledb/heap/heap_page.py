"""
Heap Page class representing a page full of records.
"""

from simpledb.disk.data_page import DataPage
from simpledb.disk.page import Page
from simpledb.main.catalog.tuple_desc import TupleDesc


class HeapPage(DataPage):
    """Represents a page full of records."""

    def __init__(self, page: Page, schema: TupleDesc):
        """Initialize a HeapPage."""
        self.data = page.get_data()
        self.schema = schema

    def iterator(self):
        """Get an iterator over tuples in this page."""
        from simpledb.access.read.data_page_iterator import DataPageIterator
        return DataPageIterator(self, self.schema)
