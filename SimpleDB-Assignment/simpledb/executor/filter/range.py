"""
Range predicates for filtering tuples.

Filters records based on value ranges using comparison operators.
Note: Not compatible with Boolean schema values.
"""

from abc import abstractmethod
from simpledb.executor.filter.filter import Predicate
from simpledb.heap.tuple import Tuple
from simpledb.main.catalog.type import Type


class UnsupportedOperationError(Exception):
    """Raised when match condition not supported with certain data types (e.g. BOOLEAN)."""
    pass


class Range(Predicate):
    """Abstract base class for range-based predicates."""

    def __init__(self, range_value, sorted_data=False):
        """
        Initialize a Range predicate.
        
        Args:
            range_value: The value to compare against
            sorted_data: Whether the data is sorted (for optimization)
        """
        self.range_value = range_value
        self.sorted_data = sorted_data

    @abstractmethod
    def matches(self, compared_value: int) -> bool:
        """
        Determine if a comparison result matches the range condition.
        
        Args:
            compared_value: Result of comparison (-1, 0, or 1)
            
        Returns:
            True if the value matches the range condition
        """
        pass

    def matches_condition(self, row: Tuple, column_name: str) -> bool:
        """Check if the column value is within the range."""
        index = row.get_schema().get_index_from_name(column_name)
        column_type = row.get_schema().get_field_type(index)
        column_value = row.get_column(index)

        if column_type == Type.STRING:
            compared = self._compare_strings(column_value, self.range_value)
        elif column_type == Type.INTEGER:
            compared = self._compare_integers(column_value, self.range_value)
        elif column_type == Type.DOUBLE:
            compared = self._compare_doubles(column_value, self.range_value)
        elif column_type == Type.BOOLEAN:
            raise UnsupportedOperationError("Range predicates are not supported for BOOLEAN types")
        else:
            raise UnsupportedOperationError(f"Unsupported type for range comparison: {column_type}")

        return self.matches(compared)

    def should_stop_searching(self, row: Tuple, column_name: str) -> bool:
        """Never stop searching for range predicates."""
        return False

    @staticmethod
    def _compare_strings(a: str, b: str) -> int:
        """Compare two strings. Returns -1, 0, or 1."""
        if a < b:
            return -1
        elif a > b:
            return 1
        else:
            return 0

    @staticmethod
    def _compare_integers(a: int, b: int) -> int:
        """Compare two integers. Returns -1, 0, or 1."""
        if a < b:
            return -1
        elif a > b:
            return 1
        else:
            return 0

    @staticmethod
    def _compare_doubles(a: float, b: float) -> int:
        """Compare two doubles. Returns -1, 0, or 1."""
        if a < b:
            return -1
        elif a > b:
            return 1
        else:
            return 0


class GreaterThanEquals(Range):
    """Predicate for >= comparisons."""

    def matches(self, compared_value: int) -> bool:
        """Check if compared_value >= 0."""
        return compared_value >= 0


class GreaterThan(Range):
    """Predicate for > comparisons."""

    def matches(self, compared_value: int) -> bool:
        """Check if compared_value > 0."""
        return compared_value > 0


class LessThanEquals(Range):
    """Predicate for <= comparisons."""

    def matches(self, compared_value: int) -> bool:
        """Check if compared_value <= 0."""
        return compared_value <= 0


class LessThan(Range):
    """Predicate for < comparisons."""

    def matches(self, compared_value: int) -> bool:
        """Check if compared_value < 0."""
        return compared_value < 0
