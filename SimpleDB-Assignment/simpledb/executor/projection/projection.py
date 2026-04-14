"""
Projection operator for filtering columns.
"""

from simpledb.access.read.access_iterator import AccessIterator
from simpledb.main.catalog.tuple_desc import TupleDesc
from simpledb.heap.tuple import Tuple


class Projection(AccessIterator):
    """Implements column projection over an iterator."""

    def __init__(self, child: AccessIterator, *projected_columns: str):
        """Initialize Projection."""
        self.child = child
        self.parent_schema = child.get_schema()
        self.projected_columns = projected_columns
        self.schema = self.parent_schema.project(*projected_columns)

    def close(self) -> None:
        """Close the projection."""
        self.child.close()

    def get_schema(self) -> TupleDesc:
        """Get the schema of projected tuples."""
        return self.schema

    def has_next(self) -> bool:
        """Check if there is a next tuple."""
        return self.child.has_next()

    def __next__(self) -> Tuple:
        """Get the next projected tuple."""
        parent_tuple = self.child.__next__()
        projected = Tuple(self.schema)
        for i in range(self.schema.get_num_fields()):
            field_name = self.schema.get_field_name(i)
            projected.set_column(i, parent_tuple.get_column(field_name))
        return projected

    def __iter__(self):
        """Return self as iterator."""
        return self

    def mark(self) -> None:
        """Mark the current position."""
        self.child.mark()

    def reset(self) -> None:
        """Reset to the marked position."""
        self.child.reset()
