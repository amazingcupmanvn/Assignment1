"""
NotModifier predicate for inverting other predicates.

Allows you to negate other predicates, effectively implementing NOT logic.
"""

from simpledb.executor.filter.filter import Predicate
from simpledb.heap.tuple import Tuple


class NotModifier(Predicate):
    """Predicate that inverts the result of another predicate."""

    def __init__(self, original_predicate: Predicate):
        """
        Initialize a NotModifier.
        
        Args:
            original_predicate: The predicate to invert
        """
        self.predicate = original_predicate

    def matches_condition(self, row: Tuple, column_name: str) -> bool:
        """Check if the underlying predicate does NOT match."""
        return not self.predicate.matches_condition(row, column_name)

    def should_stop_searching(self, row: Tuple, column_name: str) -> bool:
        """Never stop searching for negated predicates."""
        return False
