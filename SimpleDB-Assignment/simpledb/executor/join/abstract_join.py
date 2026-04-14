"""
Abstract Join class.
"""

from abc import abstractmethod
from simpledb.access.read.access_iterator import AccessIterator
from simpledb.heap.tuple import Tuple
from simpledb.main.catalog.tuple_desc import TupleDesc


class AbstractJoin(AccessIterator):
    """Abstract base class for join implementations."""

    def __init__(self, left: AccessIterator, right: AccessIterator, condition):
        """Initialize the AbstractJoin."""
        self.left = left
        self.right = right
        self.schema = TupleDesc.join(left.get_schema(), right.get_schema())
        self.left_column = left.get_schema().get_index_from_name(condition.get_left_column())
        self.right_column = right.get_schema().get_index_from_name(condition.get_right_column())

    def join_tuple(self, left_tuple: Tuple, right_tuple: Tuple) -> Tuple:
        """Join two tuples together."""
        tuple_obj = Tuple(self.get_schema())
        tuple_obj.copy_values(left_tuple)
        tuple_obj.copy_values(right_tuple)
        return tuple_obj

    def close(self) -> None:
        """Close the join."""
        self.left.close()
        self.right.close()

    def get_schema(self) -> TupleDesc:
        """Get the schema of joined tuples."""
        return self.schema

    @abstractmethod
    def has_next(self) -> bool:
        """Check if there is a next joined tuple."""
        pass

    @abstractmethod
    def __next__(self) -> Tuple:
        """Get the next joined tuple."""
        pass
