"""
Nested Loop Join algorithm.
"""

from simpledb.executor.join.abstract_join import AbstractJoin
from simpledb.access.read.access_iterator import AccessIterator
from simpledb.buffer.buffer_manager import BufferAccessException
from simpledb.access.read.access_iterator import UnsupportedOperationError
from simpledb.heap.tuple import Tuple


class NestedLoopJoin(AbstractJoin):
    """Implements a Nested Loop Join."""

    def __init__(self, left: AccessIterator, right: AccessIterator, condition):
        """Initialize the NestedLoopJoin."""
        super().__init__(left, right, condition)
        self.next = None
        self.current_left = None

    def has_next(self) -> bool:
        """Check if there is a next joined tuple."""
        if self.next is not None:
            return True
        
        if self.current_left is None and self.left.has_next():
            self.current_left = self.left.__next__()
        
        while self.left.has_next() or self.current_left is not None:
            if not self.right.has_next():
                if not self.left.has_next():
                    return False
                self._reset_right()
                self.current_left = self.left.__next__()
            
            current_right = self.right.__next__()
            if self.current_left.get_column(self.left_column) == current_right.get_column(self.right_column):
                self.next = self.join_tuple(self.current_left, current_right)
                return True
        
        return False

    def _reset_right(self) -> None:
        """Reset the right iterator to the beginning."""
        try:
            self.right.reset()
        except BufferAccessException as e:
            raise RuntimeError(str(e))

    def __next__(self) -> Tuple:
        """Get the next joined tuple."""
        if not self.has_next():
            raise StopIteration()
        temp = self.next
        self.next = None
        return temp

    def __iter__(self):
        """Return self as iterator."""
        return self

    def mark(self) -> None:
        """Mark operation not supported."""
        raise UnsupportedOperationError()

    def reset(self) -> None:
        """Reset operation not supported."""
        raise UnsupportedOperationError()
