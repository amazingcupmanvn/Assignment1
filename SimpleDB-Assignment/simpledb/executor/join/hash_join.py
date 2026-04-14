"""
Hash Join algorithm for equi-joins.
"""

from simpledb.executor.join.abstract_join import AbstractJoin
from simpledb.access.read.access_iterator import AccessIterator
from simpledb.access.read.access_iterator import UnsupportedOperationError
from simpledb.heap.tuple import Tuple


class HashJoin(AbstractJoin):
    """Implements an in-memory Hash Join for equi-join conditions."""

    def __init__(self, left: AccessIterator, right: AccessIterator, condition):
        super().__init__(left, right, condition)
        self.next = None
        self._hash_built = False
        self._hash_table = {}
        self._current_left = None
        self._current_matches = []
        self._match_index = 0

    def _build_hash_table(self) -> None:
        """Build a hash table over the right relation."""
        while self.right.has_next():
            right_tuple = self.right.__next__()
            key = right_tuple.get_column(self.right_column)
            self._hash_table.setdefault(key, []).append(right_tuple)
        self._hash_built = True

    def has_next(self) -> bool:
        """Check if there is a next joined tuple."""
        if self.next is not None:
            return True

        if not self._hash_built:
            self._build_hash_table()

        while True:
            # If there are remaining matches for the current left tuple, emit next one.
            if self._match_index < len(self._current_matches):
                right_tuple = self._current_matches[self._match_index]
                self._match_index += 1
                self.next = self.join_tuple(self._current_left, right_tuple)
                return True

            if not self.left.has_next():
                return False

            self._current_left = self.left.__next__()
            left_key = self._current_left.get_column(self.left_column)
            self._current_matches = self._hash_table.get(left_key, [])
            self._match_index = 0

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
