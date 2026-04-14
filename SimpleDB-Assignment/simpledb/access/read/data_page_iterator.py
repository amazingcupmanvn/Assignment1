"""
Data Page Iterator - iterates over tuples on a single page.
"""

from simpledb.disk.data_page import DataPage
from simpledb.heap.tuple import Tuple
from simpledb.main.catalog.tuple_desc import TupleDesc


class DataPageIterator:
    """Provides an iterator for the tuples on a page."""

    def __init__(self, data_page: DataPage, schema: TupleDesc):
        """Initialize the DataPageIterator."""
        self.schema = schema
        self.data_page = data_page
        self.slot_no = 0

    def __iter__(self):
        """Return self as iterator."""
        return self

    def __next__(self) -> Tuple:
        """Get the next tuple on the page."""
        if not self.has_next():
            raise StopIteration()
        new_tuple = Tuple(self.schema)
        self.data_page.get_record(self.slot_no, new_tuple)
        self.slot_no += 1
        return new_tuple

    def has_next(self) -> bool:
        """Check if there is another record on the page."""
        return self.slot_no < self.data_page.get_record_count()

    def peek_next(self) -> Tuple:
        """Get next value without progressing the iterator."""
        new_tuple = Tuple(self.schema)
        self.data_page.get_record(self.slot_no, new_tuple)
        return new_tuple

    def get_slot_no(self) -> int:
        """Get the current slot number."""
        return self.slot_no

    def set_slot(self, marked_slot: int) -> None:
        """Set the current slot."""
        self.slot_no = marked_slot
