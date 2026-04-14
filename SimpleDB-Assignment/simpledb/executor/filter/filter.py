"""
Generic Filter Class for filtering tuples based on predicates.
- Takes a Predicate to determine whether a tuple is to be selected

This class can wrap an AccessIterator to provide for pipelining of query data.
For example:
    filtered = Filter(students.iterator(), "age", Equals(20))

Will filter the entries in the student's iterator, so it will only return those 
students whose age = 20. This is essentially a way to handle the WHERE clause of a query.
The corresponding query could be written as "SELECT * FROM Students WHERE age = 20;
"""

from abc import ABC, abstractmethod
from simpledb.access.read.access_iterator import AccessIterator
from simpledb.heap.tuple import Tuple
from simpledb.main.catalog.tuple_desc import TupleDesc


class Predicate(ABC):
    """Abstract base class for filter predicates."""

    @abstractmethod
    def matches_condition(self, row: Tuple, column_name: str) -> bool:
        """
        Check if a tuple and its column match the filter condition.
        
        Args:
            row: The tuple to check
            column_name: The name of the column to check
            
        Returns:
            True if the tuple matches the condition, False otherwise
        """
        pass

    @abstractmethod
    def should_stop_searching(self, row: Tuple, column_name: str) -> bool:
        """
        Determine if we should stop searching based on the current row.
        
        This is useful for sorted data where we can stop early.
        
        Args:
            row: The current tuple being examined
            column_name: The name of the column being examined
            
        Returns:
            True if searching should stop, False otherwise
        """
        pass


class Filter(AccessIterator):
    """Filter iterator that applies a predicate to an underlying iterator."""

    def __init__(self, child_iter: AccessIterator, column_name: str, predicate: Predicate):
        """
        Initialize a Filter.
        
        Args:
            child_iter: The underlying iterator to filter
            column_name: The name of the column to filter on
            predicate: The predicate to apply
        """
        self.input = child_iter
        self.column_name = column_name
        self.predicate = predicate
        self.next_tuple = None
        self.stop = False

    def has_next(self) -> bool:
        """Check if there is a next matching tuple."""
        if self.stop:
            return False
        if self.next_tuple is not None:
            return True
        if not self.input.has_next():
            return False
        
        # Find the next matching tuple
        while self.input.has_next() and not self.stop:
            next_tuple = self.input.__next__()
            if self.predicate.matches_condition(next_tuple, self.column_name):
                self.next_tuple = next_tuple
                break
            self.stop = self.predicate.should_stop_searching(next_tuple, self.column_name)
        
        return self.next_tuple is not None

    def __next__(self) -> Tuple:
        """Get the next matching tuple."""
        if self.has_next():
            result = self.next_tuple
            self.next_tuple = None
            return result
        raise StopIteration()

    def __iter__(self):
        """Return self as iterator."""
        return self

    def get_schema(self) -> TupleDesc:
        """Get the schema of tuples produced by this iterator."""
        return self.input.get_schema()

    def close(self) -> None:
        """Close the iterator and release resources."""
        self.input.close()

    def mark(self) -> None:
        """Update the marked position to the current position."""
        self.input.mark()

    def reset(self) -> None:
        """Return to previously marked position."""
        self.input.reset()
