"""
Limit Iterator Class to restrict the number of tuples that are output.
- We will accept a tuple (match its condition), if we haven't already matched more than 'limit' tuples
- Once we have accepted 'limit' tuples, we should stop searching
"""

from simpledb.heap.tuple import Tuple
from simpledb.main.catalog.tuple_desc import TupleDesc
from simpledb.access.read.access_iterator import AccessIterator


class Limit(AccessIterator):
    """Class to restrict the number of tuples that are output."""

    def __init__(self, child_iter: AccessIterator, limit: int):
        """Initialize the Limit iterator.

        Args:
            child_iter: The underlying iterator to limit
            limit: The maximum number of tuples to return
        """
        self.input = child_iter
        self.count = 0
        self.limit = limit

    def close(self) -> None:
        """Close the iterator and release resources."""
        self.input.close()

    def get_schema(self) -> TupleDesc:
        """Get the schema of tuples produced by this iterator."""
        return self.input.get_schema()

    def has_next(self) -> bool:
        """Check if there is a next tuple."""
        return self.input.has_next() and self.count < self.limit

    def __next__(self) -> Tuple:
        """Get the next tuple."""
        if not self.has_next():
            raise StopIteration()
        self.count += 1
        return self.input.__next__()

    def __iter__(self):
        """Return self as iterator."""
        return self

    def mark(self) -> None:
        """Update the marked position to the current position."""
        self.input.mark()

    def reset(self) -> None:
        """Return to previously marked position."""
        self.input.reset()
