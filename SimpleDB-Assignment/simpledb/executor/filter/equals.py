"""
Equality predicate for filtering tuples.

Implements equality testing to check if a column value equals a target value.
"""

from simpledb.executor.filter.filter import Predicate
from simpledb.heap.tuple import Tuple


class Equals(Predicate):
    """Predicate that checks if a column value equals a target value."""

    def __init__(self, target_value):
        """
        Initialize an Equals predicate.
        
        Args:
            target_value: The value to compare column values against
        """
        self.value = target_value

    def matches_condition(self, row: Tuple, column_name: str) -> bool:
        """Check if the column value equals the target value."""
        index = row.get_schema().get_index_from_name(column_name)
        return row.get_column(index) == self.value

    def should_stop_searching(self, row: Tuple, column_name: str) -> bool:
        """Never stop searching for equality predicates."""
        return False
