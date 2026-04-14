"""
Generic Iterator Class for Access Patterns.
"""

from abc import ABC, abstractmethod
from typing import Iterator
from simpledb.heap.tuple import Tuple
from simpledb.main.catalog.tuple_desc import TupleDesc


class UnsupportedOperationError(Exception):
    """Raised when trying to execute an unsupported operation on an iterator."""
    pass


class AccessIterator(ABC, Iterator):
    """Generic Iterator Class for database access patterns."""

    def __iter__(self):
        """Return the iterator object itself to support Python iterable."""
        return self

    @abstractmethod
    def __next__(self) -> Tuple:
        """Get the next tuple to support Python iterable."""
        pass

    @abstractmethod
    def has_next(self) -> bool:
        """Check if there is a next element."""
        pass

    @abstractmethod
    def get_schema(self) -> TupleDesc:
        """Get the schema of tuples produced by this iterator."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the iterator and release resources."""
        pass

    @abstractmethod
    def mark(self) -> None:
        """Update the marked position to the current position."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Return to previously marked position."""
        pass
