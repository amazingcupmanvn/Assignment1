"""
Data File Iterator base class.
"""

from abc import ABC, abstractmethod
from simpledb.access.read.access_iterator import AccessIterator
from simpledb.buffer.buffer_manager import BufferAccessException, BufferManager
from simpledb.main.catalog.tuple_desc import TupleDesc
from simpledb.heap.page_id import PageId
from simpledb.heap.tuple import Tuple
from simpledb.disk.data_page import DataPage


class DataFileIterator(AccessIterator, ABC):
    """Iterator to traverse a file collection of data pages."""

    def __init__(self, buffer_manager: BufferManager, data_file_page: PageId, schema: TupleDesc):
        """Initialize the DataFileIterator."""
        self.buffer_manager = buffer_manager
        self.schema = schema
        self.current_page_id = PageId(data_file_page.get())
        self.marked_page_id  = None
        self.marked_slot     = None
        self.page_iterator   = None
        self.current_data_page = None
        
        try:
            self._advance_to_next_page()
            self.mark()
        except BufferAccessException:
            pass

    def _advance_to_next_page(self, reset_to_mark: bool = False) -> None:
        """Advance to the next page."""
        if self.current_data_page is not None:
            next_page_id = self.current_data_page.get_next_page()
            self.buffer_manager.unpin(self.current_page_id, False)
            if reset_to_mark:
                self.current_page_id = self.marked_page_id
            elif next_page_id.is_valid():
                self.current_page_id = next_page_id
            else:
                self.page_iterator = None
                return
        
        self.current_data_page = self.get_data_page(self.current_page_id)
        self.page_iterator = self.current_data_page.iterator()

    @abstractmethod
    def get_data_page(self, page_id: PageId) -> DataPage:
        """Get the data page for the given page ID."""
        pass

    def __iter__(self):
        """Return self as iterator."""
        return self

    def __next__(self) -> Tuple:
        """Get the next tuple."""
        if not self.has_next():
            raise StopIteration()

        return self.page_iterator.__next__()

    def has_next(self) -> bool:
        """Check if there is a next tuple."""
        if self.page_iterator is None:
            return False
        
        if self.page_iterator.has_next():
            return True
        
        # Try next page
        if self.current_page_id.is_valid():
            try:
                self._advance_to_next_page()
                return self.has_next()
            except:
                return False
        
        return False

    def get_schema(self) -> TupleDesc:
        """Get the schema of tuples."""
        return self.schema

    def close(self) -> None:
        """Close the iterator."""
        if self.current_data_page is not None:
            self.buffer_manager.unpin(self.current_page_id, False)

    def mark(self) -> None:
        """Mark the current position."""
        self.marked_page_id = PageId(self.current_page_id.get())
        if self.page_iterator is not None:
            self.marked_slot = self.page_iterator.get_slot_no()

    def reset(self) -> None:
        """Reset to the marked position."""
        if self.marked_page_id is not None:
            try:
                self._advance_to_next_page(True)
                if self.page_iterator is not None:
                    self.page_iterator.set_slot(self.marked_slot)
            except BufferAccessException:
                pass
        else:
            raise RuntimeError()